use polars::prelude::{
    Column, DataFrame, DataType, IntoLazy, JoinType, MaintainOrderJoin, PolarsResult,
    UniqueKeepStrategy, all_horizontal, col, dtype_cols,
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
