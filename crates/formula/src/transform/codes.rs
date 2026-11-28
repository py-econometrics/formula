use polars::prelude::{
    ChunkCompareEq, Column, DataFrame, DataType, IntoColumn, IntoLazy, IntoSeries, JoinType,
    MaintainOrderJoin, NULL, PlSmallStr, PolarsResult, Scalar, Series, UniqueKeepStrategy,
    all_horizontal, col, dtype_cols, lit, polars_ensure, when,
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

/// The position of the column among `columns` that holds each of `level_count` levels
///
/// Each column lists the positions of the levels it holds.
pub(super) fn level_columns<'a>(
    columns: impl IntoIterator<Item = &'a [usize]>,
    level_count: usize,
) -> Vec<usize> {
    let mut level_columns = vec![0; level_count];
    for (column, levels) in columns.into_iter().enumerate() {
        for &level in levels {
            level_columns[level] = column;
        }
    }
    level_columns
}

pub(super) fn fitted_levels(levels: &Series) -> PolarsResult<Vec<Scalar>> {
    (0..levels.len())
        .map(|index| {
            Ok(Scalar::new(
                levels.dtype().clone(),
                levels.get(index)?.into_static(),
            ))
        })
        .collect()
}

/// Each row's code from a categorical transform, and the number of kept and omitted codes
///
/// Codes `0..k` have indicator columns, where `k` is the number of kept codes, and the remaining
/// codes up to `count()` were omitted. A row whose code was omitted is false in every indicator
/// column, and a row whose code is null is null in every indicator column.
#[derive(Clone, Debug)]
pub(crate) struct Codes {
    codes: Column,
    count: u32,
}

impl Codes {
    /// Each row's code: the position in `columns` of the column that holds the row's value
    ///
    /// Each column lists the positions in `levels` of the values it holds. A null, or a value
    /// that isn't one of `levels`, has a null code.
    pub(super) fn new<'a>(
        input: &Column,
        levels: &Series,
        columns: impl ExactSizeIterator<Item = &'a [usize]>,
    ) -> PolarsResult<Self> {
        let count = columns.len() as u32;
        let level_codes = level_columns(columns, levels.len())
            .into_iter()
            .map(|column| column as u32)
            .collect::<Vec<_>>();
        // For a small number of levels comparing each row with each level is cheaper than a join's
        // single hash lookup per row; there's no hard evidence for choosing 8
        let codes = if level_codes.len() <= 8 {
            let argument = col(input.name().clone());
            let expression = fitted_levels(levels)?
                .into_iter()
                .zip(&level_codes)
                .rev()
                .fold(
                    lit(NULL).cast(DataType::UInt32),
                    |otherwise, (level, code)| {
                        when(argument.clone().eq(lit(level)))
                            .then(lit(*code))
                            .otherwise(otherwise)
                    },
                );
            let frame = input
                .clone()
                .into_frame()
                .lazy()
                .with_column(expression.alias(CODE_COLUMN))
                .collect()?;
            frame.column(CODE_COLUMN)?.clone()
        } else {
            let mapping = DataFrame::new(
                levels.len(),
                vec![
                    levels.clone().with_name(input.name().clone()).into_column(),
                    Column::new(CODE_COLUMN.into(), level_codes),
                ],
            )?;
            map_codes(&input.clone().into_frame(), &mapping)?
        };
        Ok(Self { codes, count })
    }

    pub(super) fn indicators(
        &self,
        names: impl IntoIterator<Item = PlSmallStr>,
    ) -> PolarsResult<DataFrame> {
        let indicators = names
            .into_iter()
            .enumerate()
            .map(|(code, name)| {
                let indicator = self
                    .codes
                    .as_materialized_series()
                    .equal(code as u32)?
                    .with_name(name);
                Ok(indicator.into_column())
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        DataFrame::new(self.codes.len(), indicators)
    }

    pub(crate) fn codes(&self) -> &Column {
        &self.codes
    }

    pub(crate) fn count(&self) -> u32 {
        self.count
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
    use polars::prelude::NamedFrom;

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
        let input = Column::new("x".into(), [Some("a"), Some("b"), Some("c"), None]);
        let levels = Series::new("x".into(), ["a", "b", "c"]);
        let columns: [&[usize]; 3] = [&[0], &[1], &[2]];
        let codes = Codes::new(&input, &levels, columns.into_iter()).unwrap();
        let indicators = codes.indicators(["a", "b"].map(PlSmallStr::from)).unwrap();
        assert_eq!(codes.count(), 3);
        let expected = polars::df!(
            "a" => [Some(true), Some(false), Some(false), None],
            "b" => [Some(false), Some(true), Some(false), None],
        )
        .unwrap();
        assert_eq!(indicators.width(), 2);
        assert!(indicators.equals_missing(&expected));
    }
}
