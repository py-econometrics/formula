use crate::formula::{ComponentKey, MaterializedColumns};
use polars::prelude::{
    ChunkCompareEq, Column, DataFrame, DataType, IntoColumn, IntoLazy, IntoSeries, JoinType,
    MaintainOrderJoin, NULL, PlSmallStr, PolarsResult, Series, UniqueKeepStrategy, all_horizontal,
    col, dtype_cols, lit, polars_ensure, when,
};

/// The column of a mapping frame that holds each tuple's code
pub(super) const CODE_COLUMN: &str = "__code";

/// Numbers the distinct tuples of values in `frame`
///
/// The result has one row per tuple: `frame`'s columns, and the tuple's code in `CODE_COLUMN`.
/// Tuples containing a null or a nonfinite float are omitted.
pub(super) fn tuple_codes(frame: &DataFrame) -> PolarsResult<DataFrame> {
    // Prevent encoding nonfinite floats as a category
    let finite = dtype_cols([DataType::Float32, DataType::Float64])
        .as_selector()
        .as_expr()
        .is_finite();
    frame
        .clone()
        .lazy()
        .drop_nulls(None)
        .filter(all_horizontal([finite])?)
        .unique_stable(None, UniqueKeepStrategy::First)
        .with_row_index(CODE_COLUMN, None)
        .with_columns([col(CODE_COLUMN).strict_cast(DataType::UInt32)])
        .collect()
}

pub(super) fn map_codes(frame: &DataFrame, mapping: &DataFrame) -> PolarsResult<Column> {
    let keys = frame
        .columns()
        .iter()
        .map(|column| col(column.name().clone()))
        .collect::<Vec<_>>();
    let codes = frame
        .clone()
        .lazy()
        .join_builder()
        .with(mapping.clone().lazy())
        .how(JoinType::Left)
        .on(keys)
        .join_nulls(false)
        .maintain_order(MaintainOrderJoin::Left)
        .finish()
        .select([col(CODE_COLUMN)])
        .collect()?;
    Ok(codes.column(CODE_COLUMN)?.clone())
}

/// Each row's code, and an indicator column for each kept code
///
/// Codes `0..kept()` are kept, and indicator column `i` is true where a row's code is `i`. Codes
/// `kept()..count()` were omitted, so their rows are `false` in every column.
#[derive(Clone, Debug)]
pub(crate) struct Indicators {
    columns: MaterializedColumns<ComponentKey>,
    codes: Column,
    omitted: u32,
}

impl Indicators {
    /// Builds indicator column `i` with the name and key `columns` gives at position `i`
    ///
    /// Each indicator column is true where a row's code is its position, and null where the code
    /// is null. The `omitted` codes that follow the last column's position have no column.
    pub(super) fn new(
        codes: Column,
        columns: impl IntoIterator<Item = (PlSmallStr, ComponentKey)>,
        omitted: u32,
    ) -> PolarsResult<Self> {
        let mut indicators = Vec::new();
        let mut keys = Vec::new();
        for (code, (name, key)) in columns.into_iter().enumerate() {
            let indicator = codes
                .as_materialized_series()
                .equal(code as u32)?
                .with_name(name);
            indicators.push(indicator.into_column());
            keys.push(key);
        }
        let indicators = Self {
            columns: MaterializedColumns::new(DataFrame::new(codes.len(), indicators)?, keys),
            codes,
            omitted,
        };
        let count = indicators.count();
        // Only debug builds check, as scanning every row's code costs about as much as
        // transforming a factor with few levels
        debug_assert!(
            indicators
                .codes
                .as_materialized_series()
                .max::<u32>()?
                .is_none_or(|code| code < count),
            "each code must be less than the number of kept and omitted codes",
        );
        Ok(indicators)
    }

    pub(crate) fn columns(&self) -> &MaterializedColumns<ComponentKey> {
        &self.columns
    }

    pub(crate) fn codes(&self) -> &Column {
        &self.codes
    }

    /// The number of kept codes (equals number of indicator columns)
    pub(crate) fn kept(&self) -> u32 {
        self.columns.frame().width() as u32
    }

    /// The number of kept and omitted codes
    pub(crate) fn count(&self) -> u32 {
        self.kept() + self.omitted
    }
}

/// The combinations of codes observd during fitting
#[derive(Clone, Debug)]
pub(crate) struct Combinations {
    /// The number of possible codes in each column of codes
    counts: Vec<u32>,
    /// The observed joint codes
    observed: Series,
}

impl Combinations {
    /// Records the combinations in the rows of `codes` that have no null code
    ///
    /// Each value in `codes[k]` is less than `counts[k]`.
    pub(crate) fn fit(codes: &[&Column], counts: Vec<u32>) -> PolarsResult<Self> {
        let observed = Self::combine_codes(codes, &counts)?
            .with_name(CODE_COLUMN.into())
            .drop_nulls()
            .unique()?
            .take_materialized_series();
        Ok(Self { counts, observed })
    }

    fn combine_codes(codes: &[&Column], counts: &[u32]) -> PolarsResult<Column> {
        let total = counts
            .iter()
            .try_fold(1_u64, |total, count| total.checked_mul(u64::from(*count)));
        polars_ensure!(total.is_some(), ComputeError: "too many combinations of codes");
        let mut combined = codes[0].cast(&DataType::UInt64)?;
        for (code, count) in codes.iter().zip(counts).skip(1) {
            combined = ((&combined * u64::from(*count)) + code.cast(&DataType::UInt64)?)?;
        }
        Ok(combined)
    }

    pub(crate) fn observed(&self, kept_counts: &[u32]) -> PolarsResult<Vec<(Vec<u32>, u64)>> {
        let mut combinations = Vec::new();
        let mut codes = vec![0; self.counts.len()];

        'combination: for joint_code in self.observed.u64()?.into_no_null_iter() {
            let mut remaining = joint_code;

            for (index, &count) in self.counts.iter().enumerate().rev() {
                let count = u64::from(count);
                let code = (remaining % count) as u32;
                if code >= kept_counts[index] {
                    continue 'combination;
                }

                codes[index] = code;
                remaining /= count;
            }

            combinations.push((codes.clone(), joint_code));
        }

        Ok(combinations)
    }

    /// Each row's combination code, null if any of the row's codes is null or its combination
    /// wasn't observed when fitting
    pub(crate) fn transform(&self, codes: &[&Column]) -> PolarsResult<Column> {
        let combined = Self::combine_codes(codes, &self.counts)?.with_name(CODE_COLUMN.into());
        let argument = col(CODE_COLUMN);
        let observed = argument
            .clone()
            .is_in(lit(self.observed.implode()?.into_series()), false);
        let frame = combined
            .into_frame()
            .lazy()
            .select([when(observed)
                .then(argument)
                .otherwise(lit(NULL))
                .alias(CODE_COLUMN)])
            .collect()?;
        Ok(frame.column(CODE_COLUMN)?.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_decodes_full_width_unsigned_codes() {
        let count = u32::MAX;
        let first = Column::new("first".into(), [count - 1, 1, count - 2]);
        let second = Column::new("second".into(), [count - 2, 2, count - 3]);
        let combinations = Combinations::fit(&[&first, &second], vec![count, count]).unwrap();
        let mut actual = combinations.observed(&[count, count]).unwrap();
        actual.sort_unstable();
        let expected =
            [(1, 2), (count - 2, count - 3), (count - 1, count - 2)].map(|(left, right)| {
                (
                    vec![left, right],
                    u64::from(left) * u64::from(count) + u64::from(right),
                )
            });
        assert_eq!(actual, expected);
        assert_eq!(
            combinations.observed(&[count - 2, count]).unwrap(),
            [expected[0].clone()]
        );
    }

    #[test]
    fn observed_is_empty_when_a_factor_has_no_levels() {
        let first = Column::new("first".into(), [None::<u32>, None]);
        let second = Column::new("second".into(), [Some(0_u32), None]);
        let combinations = Combinations::fit(&[&first, &second], vec![0, 1]).unwrap();
        assert!(combinations.observed(&[0, 1]).unwrap().is_empty());
    }

    #[test]
    fn indicators_mark_kept_codes_and_leave_omitted_codes_false() {
        let codes = Column::new("codes".into(), [Some(0_u32), Some(1), Some(2), None]);
        let columns = ["a", "b"].map(|name| (PlSmallStr::from(name), ComponentKey::None));
        let indicators = Indicators::new(codes, columns, 1).unwrap();
        assert_eq!((indicators.kept(), indicators.count()), (2, 3));
        let expected = polars::df!(
            "a" => [Some(true), Some(false), Some(false), None],
            "b" => [Some(false), Some(true), Some(false), None],
        )
        .unwrap();
        assert!(indicators.columns().frame().equals_missing(&expected));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "each code must be less than the number of kept and omitted codes")]
    fn indicators_reject_codes_beyond_the_omitted_ones() {
        let codes = Column::new("codes".into(), [0_u32, 2]);
        let columns = [(PlSmallStr::from("a"), ComponentKey::None)];
        let _ = Indicators::new(codes, columns, 1);
    }
}
