use polars::prelude::{
    Column, DataFrame, DataType, IntoLazy, IntoSeries, JoinType, MaintainOrderJoin, NULL,
    PolarsResult, Series, UniqueKeepStrategy, all_horizontal, col, dtype_cols, lit, polars_ensure,
    when,
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

/// Each row's code in `mapping`, null if its tuple of values has none
///
/// `mapping` has a column named like each of `frame`'s, and the codes in `CODE_COLUMN`.
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

/// Each row's combination of `codes` as one code, null if any of the row's codes is null
///
/// Each value in `codes[k]` is less than `counts[k]`. A combination's code is its position when all
/// combinations are listed with `codes[0]` varying slowest.
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

/// The combinations of codes observed when fitting
#[derive(Clone, Debug)]
pub(crate) struct Combinations {
    /// The number of possible codes in each column of codes
    counts: Vec<u32>,
    /// The codes of the observed combinations
    observed: Series,
}

impl Combinations {
    /// Records the combinations in the rows of `codes` that have no null code
    ///
    /// Each value in `codes[k]` is less than `counts[k]`.
    pub(crate) fn fit(codes: &[&Column], counts: Vec<u32>) -> PolarsResult<Self> {
        let observed = combine_codes(codes, &counts)?
            .drop_nulls()
            .unique()?
            .take_materialized_series();
        Ok(Self { counts, observed })
    }

    /// Each row's combination code, null if any of the row's codes is null or its combination
    /// wasn't observed when fitting
    pub(crate) fn transform(&self, codes: &[&Column]) -> PolarsResult<Column> {
        let combined = combine_codes(codes, &self.counts)?.with_name(CODE_COLUMN.into());
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
