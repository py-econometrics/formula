use crate::SelectionError;
use crate::fit::{Definition, Fit};
use crate::model::{FittedFormula, FittedFormulas, ModelError};
use crate::parse::{
    FORMULA_SEPARATOR, PIPE_SEPARATOR, ParseResult, is_identifier_character, parse_scalar,
};
use crate::transform::{
    ArgumentCount, IntegerEncodingDefinition, Transform, TransformError, validate_numeric_input,
};
use polars::prelude::{AnyValue, Column, DataFrame, DataType, PolarsResult, Scalar};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;
use winnow::Parser;
use winnow::combinator::fail;
use winnow::error::{ContextError, StrContext, StrContextValue};

pub(crate) fn push_unique<T: PartialEq>(items: &mut Vec<T>, item: T) {
    if !items.contains(&item) {
        items.push(item);
    }
}

fn collect_unique<T: PartialEq>(items: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut unique = Vec::new();
    for item in items {
        push_unique(&mut unique, item);
    }
    unique
}

pub(crate) fn render_column_name(name: &str) -> String {
    let can_be_bare = !name.is_empty()
        && name.chars().all(is_identifier_character)
        && parse_scalar.parse(name).is_err();

    if can_be_bare {
        name.to_owned()
    } else {
        format!("`{}`", name.replace('`', "``"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Dependent,
    Independent,
    FixedEffect,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VariableDefinition {
    name: String,
}

impl VariableDefinition {
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug)]
pub(crate) struct VariableState {
    dtype: DataType,
}

impl Definition<VariableState> for VariableDefinition {
    type Input = Column;
    type Output = Column;
    type Error = TransformError;

    fn fit(&self, input: &Column) -> Result<VariableState, TransformError> {
        Ok(VariableState {
            dtype: input.dtype().clone(),
        })
    }

    fn validate(&self, state: &VariableState, input: &Column) -> Result<(), TransformError> {
        if input.dtype() != &state.dtype {
            return Err(TransformError::InputTypeMismatch {
                name: input.name().clone(),
                expected: state.dtype.to_string(),
                actual: input.dtype().to_string(),
            });
        }

        Ok(())
    }

    fn transform(&self, _state: &VariableState, input: &Column) -> Result<Column, TransformError> {
        Ok(input.clone())
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum Factor {
    Literal(Scalar),
    Variable(Fit<VariableDefinition, VariableState>),
    Call {
        transform: Transform,
        arguments: Vec<Arc<Factor>>,
    },
}

impl From<&str> for Factor {
    fn from(value: &str) -> Self {
        Self::Variable(Fit::Unfitted(VariableDefinition {
            name: value.to_owned(),
        }))
    }
}

impl Factor {
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Variable(_) | Self::Literal(_) => Ok(()),
            Self::Call {
                transform,
                arguments,
            } => {
                let expected = transform.argument_count();

                if !expected.accepts(arguments.len()) {
                    return Err(ValidationError::InvalidArgumentCount {
                        call: self.render(),
                        expected,
                        actual: arguments.len(),
                    });
                }

                for argument in arguments {
                    argument.validate()?;
                    if let Self::Call { transform, .. } = argument.as_ref()
                        && transform.expands()
                    {
                        return Err(ValidationError::InvalidTransformComposition);
                    }
                }

                Ok(())
            }
        }
    }

    fn render(&self) -> String {
        match self {
            Self::Variable(variable) => render_column_name(&variable.definition().name),
            Self::Literal(value) => match value.as_any_value() {
                AnyValue::Int64(value) => value.to_string(),
                AnyValue::Float64(value) => format!("{value:?}"),
                _ => unreachable!("formula arguments contain only numeric literals"),
            },
            Self::Call {
                transform,
                arguments,
            } => {
                let arguments = arguments
                    .iter()
                    .map(|argument| argument.render())
                    .collect::<Vec<_>>();
                let names = arguments.iter().map(String::as_str).collect::<Vec<_>>();
                transform.call_name(&names).to_string()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Term {
    factors: Vec<Arc<Factor>>,
}

// Interaction identity ignores factor order: a:b equals b:a.
// Keep storage order for output naming and column ordering.
// This comparison relies on terms containing no duplicate factors.
impl PartialEq for Term {
    fn eq(&self, other: &Self) -> bool {
        self.factors.len() == other.factors.len()
            && self
                .factors
                .iter()
                .all(|factor| other.factors.contains(factor))
    }
}

impl FromIterator<Arc<Factor>> for Term {
    fn from_iter<T: IntoIterator<Item = Arc<Factor>>>(factors: T) -> Self {
        Self {
            factors: collect_unique(factors),
        }
    }
}

impl FromIterator<Factor> for Term {
    fn from_iter<T: IntoIterator<Item = Factor>>(factors: T) -> Self {
        factors.into_iter().map(Arc::new).collect()
    }
}

impl From<Factor> for Term {
    fn from(factor: Factor) -> Self {
        std::iter::once(factor).collect()
    }
}

impl Term {
    const INTERCEPT_COLUMN_NAME: &'static str = "Intercept";

    pub(crate) fn intercept() -> Self {
        Self { factors: vec![] }
    }

    pub(crate) fn as_factor(&self) -> Option<&Factor> {
        match self.factors.as_slice() {
            [factor] => Some(factor.as_ref()),
            _ => None,
        }
    }

    fn is_intercept(&self) -> bool {
        self.factors.is_empty()
    }

    fn render(&self) -> String {
        if self.is_intercept() {
            return "1".to_owned();
        }

        self.factors
            .iter()
            .map(|factor| factor.render())
            .collect::<Vec<_>>()
            .join(":")
    }

    fn interact(&self, right: &Term) -> Term {
        self.factors
            .iter()
            .chain(right.factors.iter())
            .cloned()
            .collect()
    }

    fn validate(&self) -> Result<(), ValidationError> {
        for factor in &self.factors {
            factor.validate()?;
        }
        Ok(())
    }

    fn assemble_columns(
        &self,
        height: usize,
        factor_columns: Vec<MaterializedColumns>,
    ) -> Result<MaterializedColumns, TransformError> {
        if self.is_intercept() {
            let column = Column::new_scalar(
                Self::INTERCEPT_COLUMN_NAME.into(),
                Scalar::from(1.0_f64),
                height,
            );
            return Ok(MaterializedColumns::single_column(
                column.into_frame(),
                ColumnMetadata { keys: vec![] },
            ));
        }
        let mut column_groups = factor_columns.into_iter();
        let mut columns = column_groups
            .next()
            .expect("non-intercept terms contain at least one factor");
        for right_columns in column_groups {
            let left_columns = columns
                .iter()
                .map(|(column, metadata)| Ok((column.cast(&DataType::Float64)?, metadata)))
                .collect::<PolarsResult<Vec<_>>>()?;

            let right_columns = right_columns
                .iter()
                .map(|(column, metadata)| Ok((column.cast(&DataType::Float64)?, metadata)))
                .collect::<PolarsResult<Vec<_>>>()?;

            let mut interactions = Vec::new();
            let mut metadata = Vec::new();
            for (left_column, left_metadata) in &left_columns {
                for (right_column, right_metadata) in &right_columns {
                    // Combine columns
                    interactions.push((left_column * right_column)?.with_name(
                        format!("{}:{}", left_column.name(), right_column.name()).into(),
                    ));
                    // Combine metadata
                    metadata.push(ColumnMetadata {
                        keys: left_metadata
                            .keys
                            .iter()
                            .chain(&right_metadata.keys)
                            .cloned()
                            .collect(),
                    });
                }
            }

            columns = MaterializedColumns::new(DataFrame::new(height, interactions)?, metadata);
        }

        Ok(columns)
    }

    fn assemble(
        &self,
        height: usize,
        transformed: &[FittedFactor],
    ) -> Result<(Self, MaterializedColumns), TransformError> {
        let mut factors = Vec::with_capacity(self.factors.len());
        let mut factor_columns = Vec::with_capacity(self.factors.len());
        for factor in &self.factors {
            let result = lookup_factor(transformed, factor);
            if let Factor::Variable(_) = factor.as_ref() {
                let [column] = result.columns.frame.columns() else {
                    unreachable!("a variable transforms to exactly one column");
                };
                validate_numeric_input(column)?;
            }
            factors.push(Arc::clone(&result.factor));
            factor_columns.push(result.columns.clone());
        }

        let columns = self.assemble_columns(height, factor_columns)?;
        let term = Self { factors };

        Ok((term, columns))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Expression {
    terms: Vec<Term>,
    children: Vec<Formula>,
}

// Expression identity ignores term order.
// This comparison relies on expressions containing no duplicate terms.
impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        self.terms.len() == other.terms.len()
            && self.terms.iter().all(|term| other.terms.contains(term))
            && self.children == other.children
    }
}

impl FromIterator<Term> for Expression {
    fn from_iter<T: IntoIterator<Item = Term>>(terms: T) -> Self {
        Self::new(terms, Vec::new())
    }
}

impl From<Term> for Expression {
    fn from(term: Term) -> Self {
        Self::new(Vec::from([term]), Vec::new())
    }
}

impl Expression {
    pub(crate) fn new(terms: impl IntoIterator<Item = Term>, children: Vec<Formula>) -> Self {
        Self {
            terms: collect_unique(terms),
            children,
        }
    }

    fn render(&self) -> String {
        self.terms
            .iter()
            .map(Term::render)
            .chain(self.children.iter().map(|child| format!("[{child}]")))
            .collect::<Vec<_>>()
            .join(" + ")
    }

    fn has_intercept(&self) -> bool {
        self.terms.iter().any(Term::is_intercept)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.children.is_empty()
    }

    pub(crate) fn without_intercept(self) -> Self {
        self.subtract(Term::intercept())
    }

    fn validate(&self) -> Result<(), ValidationError> {
        for term in &self.terms {
            term.validate()?;
        }

        for child in &self.children {
            child.validate()?;
        }

        Ok(())
    }

    pub(crate) fn assemble(
        &self,
        height: usize,
        transformed: &[FittedTerm],
    ) -> Result<FittedExpression, ModelError> {
        let mut terms = Vec::with_capacity(self.terms.len());
        let mut columns = Vec::new();
        let mut metadata = Vec::new();
        let mut column_ranges = Vec::with_capacity(self.terms.len());
        for term in &self.terms {
            let entry = lookup_term(transformed, term);
            terms.push(entry.term.clone());

            let start = columns.len();
            for (column, column_metadata) in entry.columns.iter() {
                columns.push(column.clone());
                metadata.push(column_metadata.clone());
            }
            column_ranges.push(start..columns.len());
        }

        let expression = Self::new(terms, self.children.clone());
        let frame = DataFrame::new(height, columns).map_err(TransformError::from)?;
        let columns = MaterializedColumns::new(frame, metadata);
        let children = self
            .children
            .iter()
            .map(|child| child.assemble(height, transformed))
            .collect::<Result<Vec<_>, ModelError>>()?;

        Ok(FittedExpression::new(
            expression,
            columns,
            column_ranges,
            children,
        ))
    }

    pub(crate) fn add(self, other: impl Into<Self>) -> Self {
        let other = other.into();
        Self::new(
            self.terms.into_iter().chain(other.terms),
            self.children.into_iter().chain(other.children).collect(),
        )
    }
    pub(crate) fn subtract(mut self, other: impl Into<Self>) -> Self {
        let other = other.into();
        self.terms.retain(|term| !other.terms.contains(term));
        self
    }
    pub(crate) fn interact(&self, other: &Self) -> Self {
        Self::new(
            self.terms
                .iter()
                .flat_map(|left| other.terms.iter().map(move |right| left.interact(right))),
            Vec::new(),
        )
    }
    pub(crate) fn interact_fully(self, other: Self) -> Self {
        let interactions = self.interact(&other);
        self.add(other).add(interactions)
    }

    pub(crate) fn nest(self, other: Self) -> Self {
        let factors = self
            .terms
            .iter()
            .flat_map(|term| term.factors.iter().cloned())
            .collect::<Term>();

        let combined = Self::new(
            if self.terms.is_empty() {
                vec![]
            } else {
                vec![factors]
            },
            Vec::new(),
        );
        let interactions = combined.interact(&other);
        self.add(interactions)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Formula {
    pub(crate) expressions: BTreeMap<Role, Expression>,
}

impl Formula {
    pub(crate) fn validate(&self) -> Result<(), ValidationError> {
        self.expressions
            .iter()
            .try_for_each(|(_, expression)| expression.validate())
    }

    fn assemble(&self, height: usize, terms: &[FittedTerm]) -> Result<FittedFormula, ModelError> {
        let expressions = self
            .expressions
            .iter()
            .map(|(&role, expression)| {
                expression
                    .assemble(height, terms)
                    .map(|transformed| (role, transformed))
            })
            .collect::<Result<BTreeMap<_, _>, ModelError>>()?;

        Ok(FittedFormula::new(self.clone(), expressions, height))
    }

    fn collect_expressions<'a>(&'a self, output: &mut Vec<&'a Expression>) {
        for expression in self.expressions.values() {
            output.push(expression);

            for child in &expression.children {
                child.collect_expressions(output);
            }
        }
    }

    fn all_expressions(&self) -> Vec<&Expression> {
        let mut expressions = Vec::new();
        self.collect_expressions(&mut expressions);
        expressions
    }
}

impl std::fmt::Display for Formula {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dependent = &self.expressions[&Role::Dependent];
        let independent = &self.expressions[&Role::Independent];
        let fixed_effects = &self.expressions[&Role::FixedEffect];

        write!(
            f,
            "{} {FORMULA_SEPARATOR} {}",
            if dependent.is_empty() {
                "0".to_owned()
            } else {
                dependent.render()
            },
            independent.render()
        )?;

        if independent.is_empty() {
            write!(f, "-1")?;
        } else if !independent.has_intercept() {
            write!(f, " - 1")?;
        }

        if !fixed_effects.is_empty() {
            write!(
                f,
                " {PIPE_SEPARATOR} {}",
                render_fixed_effects(fixed_effects)
            )?;
        }

        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Formulas {
    formulas: Vec<Formula>,
}

impl Formulas {
    pub(crate) fn new(formulas: Vec<Formula>) -> Self {
        Self { formulas }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Formula> {
        self.formulas.iter()
    }

    pub fn len(&self) -> usize {
        self.formulas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.formulas.is_empty()
    }

    pub fn get_by_str<'a>(&self, source: &'a str) -> ParseResult<'a, Option<&Formula>> {
        let query = parse_one(source)?;
        Ok(self.iter().find(|formula| **formula == query))
    }

    pub fn fit(&self, data: &DataFrame) -> Result<FittedFormulas, ModelError> {
        for formula in &self.formulas {
            formula.validate()?;
        }
        let expressions = self
            .formulas
            .iter()
            .flat_map(Formula::all_expressions)
            .collect::<Vec<_>>();

        // Fit factors and assemble terms once across every formula
        let terms = fit_and_transform_terms(&expressions, data)?;

        let entries = self
            .formulas
            .iter()
            .map(|formula| formula.assemble(data.height(), &terms))
            .collect::<Result<Vec<_>, ModelError>>()?;

        Ok(FittedFormulas::new(entries))
    }
}

pub(crate) fn parse_one(source: &str) -> ParseResult<'_, Formula> {
    let mut formulas = crate::parse(source)?;
    if formulas.len() == 1 {
        return Ok(formulas.formulas.pop().expect("exactly one formula"));
    }

    fail::<_, _, ContextError>
        .context(StrContext::Expected(StrContextValue::Description(
            "exactly one formula",
        )))
        .parse(source)
}

impl FromIterator<Formula> for Formulas {
    fn from_iter<T: IntoIterator<Item = Formula>>(formulas: T) -> Self {
        Self::new(formulas.into_iter().collect())
    }
}

impl<'a> IntoIterator for &'a Formulas {
    type Item = &'a Formula;
    type IntoIter = std::slice::Iter<'a, Formula>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug)]
pub enum ValidationError {
    InvalidArgumentCount {
        call: String,
        expected: ArgumentCount,
        actual: usize,
    },
    InvalidTransformComposition,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ComponentKey {
    None,
    Scalar(Scalar),
    Tuple(Vec<ComponentKey>),
}

#[derive(Clone, Debug)]
pub(crate) struct ColumnMetadata {
    keys: Vec<ComponentKey>,
}

impl From<ComponentKey> for ColumnMetadata {
    fn from(key: ComponentKey) -> Self {
        Self { keys: vec![key] }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MaterializedColumns {
    frame: DataFrame,
    metadata: Vec<ColumnMetadata>,
}

impl MaterializedColumns {
    pub(crate) fn frame(&self) -> &DataFrame {
        &self.frame
    }

    pub(crate) fn new(frame: DataFrame, metadata: Vec<ColumnMetadata>) -> Self {
        assert_eq!(
            frame.width(),
            metadata.len(),
            "each materialized column must have metadata",
        );

        Self { frame, metadata }
    }

    pub(crate) fn single_column(frame: DataFrame, metadata: ColumnMetadata) -> Self {
        Self::new(frame, vec![metadata])
    }

    fn iter(&self) -> impl Iterator<Item = (&Column, &ColumnMetadata)> {
        self.frame.columns().iter().zip(&self.metadata)
    }
}

struct FittedFactor {
    factor: Arc<Factor>,
    columns: MaterializedColumns,
}

pub(crate) struct FittedTerm {
    term: Term,
    columns: MaterializedColumns,
}

#[derive(Clone, Debug)]
pub(crate) struct FittedExpression {
    expression: Expression,
    columns: MaterializedColumns,
    column_ranges: Vec<Range<usize>>,
    children: Vec<FittedFormula>,
}

impl FittedExpression {
    fn new(
        expression: Expression,
        columns: MaterializedColumns,
        column_ranges: Vec<Range<usize>>,
        children: Vec<FittedFormula>,
    ) -> Self {
        Self {
            expression,
            columns,
            column_ranges,
            children,
        }
    }

    pub(crate) fn expression(&self) -> &Expression {
        &self.expression
    }

    pub(crate) fn frame(&self) -> &DataFrame {
        &self.columns.frame
    }

    pub(crate) fn children(&self) -> &[FittedFormula] {
        &self.children
    }

    pub(crate) fn select_terms(&self, query: &Expression) -> Result<Self, SelectionError> {
        for term in &query.terms {
            if !self.expression.terms.contains(term) {
                return Err(SelectionError::MissingTerm(term.render()));
            }
        }

        let mut terms = Vec::new();
        let mut columns = Vec::new();
        let mut metadata = Vec::new();
        let mut column_ranges = Vec::new();
        for (term, range) in self.expression.terms.iter().zip(&self.column_ranges) {
            if !query.terms.contains(term) {
                continue;
            }
            terms.push(term.clone());

            let start = columns.len();
            columns.extend(self.columns.frame.columns()[range.clone()].iter().cloned());
            metadata.extend(self.columns.metadata[range.clone()].iter().cloned());
            column_ranges.push(start..columns.len());
        }

        let expression = Expression::new(terms, Vec::new());
        let frame =
            DataFrame::new(self.frame().height(), columns).map_err(SelectionError::Frame)?;

        Ok(Self::new(
            expression,
            MaterializedColumns::new(frame, metadata),
            column_ranges,
            Vec::new(),
        ))
    }
}

fn literal_column(value: &Scalar, height: usize) -> Column {
    let name = value.as_any_value().str_value().into_owned();
    Column::new_scalar(name.into(), value.clone(), height)
}

fn fit_and_transform_factors(
    scheduled: &[Arc<Factor>],
    data: &DataFrame,
) -> Result<Vec<FittedFactor>, ModelError> {
    let mut transformed = Vec::with_capacity(scheduled.len());

    for factor in scheduled {
        let result = match factor.as_ref() {
            Factor::Literal(value) => FittedFactor {
                factor: Arc::clone(factor),
                columns: MaterializedColumns::single_column(
                    literal_column(value, data.height()).into_frame(),
                    ComponentKey::None.into(),
                ),
            },
            Factor::Variable(variable) => {
                let column = data
                    .column(&variable.definition().name)
                    .map_err(TransformError::from)?;

                let fitted = variable.fit(column)?;
                let output = fitted.transform(column)?;

                FittedFactor {
                    factor: Arc::new(Factor::Variable(fitted)),
                    columns: MaterializedColumns::single_column(
                        output.into_frame(),
                        ComponentKey::None.into(),
                    ),
                }
            }
            Factor::Call {
                transform,
                arguments,
            } => {
                let mut fitted_arguments = Vec::with_capacity(arguments.len());
                let mut inputs = Vec::new();
                for argument in arguments {
                    let result = lookup_factor(&transformed, argument);
                    fitted_arguments.push(Arc::clone(&result.factor));
                    inputs.extend(result.columns.frame.columns().iter().cloned());
                }

                let transform = transform.fit(&inputs).map_err(|error| {
                    ModelError::TransformCall(format!("{}: {error}", factor.render()))
                })?;
                let output = transform.transform(&inputs).map_err(|error| {
                    ModelError::TransformCall(format!("{}: {error}", factor.render()))
                })?;

                FittedFactor {
                    factor: Arc::new(Factor::Call {
                        transform,
                        arguments: fitted_arguments,
                    }),
                    columns: output,
                }
            }
        };
        transformed.push(result)
    }
    Ok(transformed)
}

fn transform_factors(
    scheduled: &[Arc<Factor>],
    data: &DataFrame,
) -> Result<Vec<FittedFactor>, ModelError> {
    let mut transformed = Vec::with_capacity(scheduled.len());

    for factor in scheduled {
        let columns = match factor.as_ref() {
            Factor::Literal(value) => MaterializedColumns::single_column(
                literal_column(value, data.height()).into_frame(),
                ComponentKey::None.into(),
            ),
            Factor::Variable(variable) => {
                let column = data
                    .column(&variable.definition().name)
                    .map_err(TransformError::from)?;

                MaterializedColumns::single_column(
                    variable.transform(column)?.into_frame(),
                    ComponentKey::None.into(),
                )
            }
            Factor::Call {
                transform,
                arguments,
            } => {
                let mut inputs = Vec::new();
                for argument in arguments {
                    let result = lookup_factor(&transformed, argument);
                    inputs.extend(result.columns.frame.columns().iter().cloned());
                }
                transform.transform(&inputs).map_err(|error| {
                    ModelError::TransformCall(format!("{}: {error}", factor.render()))
                })?
            }
        };
        transformed.push(FittedFactor {
            factor: Arc::clone(factor),
            columns,
        });
    }
    Ok(transformed)
}

fn schedule_factors(expressions: &[&Expression]) -> Vec<Arc<Factor>> {
    fn schedule(factor: &Arc<Factor>, scheduled: &mut Vec<Arc<Factor>>) {
        // Structural equality check if factor is already scheduled.
        if scheduled.contains(factor) {
            return;
        }
        // Traverse arguments of factor call recursively.
        if let Factor::Call { arguments, .. } = factor.as_ref() {
            for argument in arguments {
                schedule(argument, scheduled);
            }
        }
        // At this point, arguments have been scheduled before the parent factor.
        scheduled.push(Arc::clone(factor));
    }

    let mut scheduled = Vec::new();
    for expression in expressions {
        for term in &expression.terms {
            for factor in &term.factors {
                schedule(factor, &mut scheduled);
            }
        }
    }

    scheduled
}

fn assemble_terms(
    expressions: &[&Expression],
    height: usize,
    factors: &[FittedFactor],
) -> Result<Vec<FittedTerm>, TransformError> {
    let mut transformed: Vec<FittedTerm> = Vec::new();

    for expression in expressions {
        for term in &expression.terms {
            if transformed.iter().any(|entry| &entry.term == term) {
                continue;
            }
            let (term, columns) = term.assemble(height, factors)?;
            transformed.push(FittedTerm { term, columns });
        }
    }

    Ok(transformed)
}

pub(crate) fn transform_terms(
    expressions: &[&Expression],
    data: &DataFrame,
) -> Result<Vec<FittedTerm>, ModelError> {
    let scheduled = schedule_factors(expressions);
    let factors = transform_factors(&scheduled, data)?;
    Ok(assemble_terms(expressions, data.height(), &factors)?)
}

fn fit_and_transform_terms(
    expressions: &[&Expression],
    data: &DataFrame,
) -> Result<Vec<FittedTerm>, ModelError> {
    let scheduled = schedule_factors(expressions);
    let factors = fit_and_transform_factors(&scheduled, data)?;
    Ok(assemble_terms(expressions, data.height(), &factors)?)
}

fn lookup_factor<'a>(transformed: &'a [FittedFactor], factor: &Arc<Factor>) -> &'a FittedFactor {
    transformed
        .iter()
        .find(|entry| &entry.factor == factor)
        .expect("factor must be transformed before its consumer")
}

fn lookup_term<'a>(transformed: &'a [FittedTerm], term: &Term) -> &'a FittedTerm {
    transformed
        .iter()
        .find(|entry| &entry.term == term)
        .expect("every term must have an transformed entry")
}

pub(crate) fn encode_fixed_effects(expression: Expression) -> Expression {
    Expression::new(
        expression
            .terms
            .into_iter()
            .filter(|term| !term.is_intercept())
            .map(|term| {
                Term::from(Factor::Call {
                    transform: Transform::F(Fit::Unfitted(IntegerEncodingDefinition)),
                    arguments: term.factors,
                })
            }),
        expression.children,
    )
}

fn render_fixed_effects(expression: &Expression) -> String {
    expression
        .terms
        .iter()
        .map(|term| {
            let [factor] = term.factors.as_slice() else {
                unreachable!("fixed-effect terms contain one encoded factor");
            };
            let Factor::Call {
                transform: Transform::F(_),
                arguments,
            } = factor.as_ref()
            else {
                unreachable!("fixed-effect terms use F encoding");
            };

            arguments
                .iter()
                .map(|argument| argument.render())
                .collect::<Vec<_>>()
                .join(":")
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgumentCount {
                call,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{call}: invalid argument count: expected {expected}, received {actual}",
                )
            }
            Self::InvalidTransformComposition => {
                f.write_str("an expanding transform cannot be used as a transform argument")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit::Fit;
    use crate::parse::parse_expression;
    use crate::transform::{
        CategoricalDefinition, ExpressionTransform, IntegerEncodingDefinition, PolynomialDefinition,
    };
    use polars::prelude::Scalar;
    use winnow::Parser;

    mod formulas {
        use crate::{Formula, Formulas, parse, parse_many};

        #[test]
        fn parsed_formulas_expose_expansion_order() {
            let formulas = parse("y ~ sw(x, z) - 1").unwrap();

            assert_eq!(formulas.len(), 2);
            assert!(!formulas.is_empty());
            assert_eq!(
                formulas.iter().map(Formula::to_string).collect::<Vec<_>>(),
                ["y ~ x - 1", "y ~ z - 1"],
            );
        }

        #[test]
        fn iteration_and_collection_preserve_order_and_duplicates() {
            let parsed = parse_many(["y ~ x - 1", "y ~ z - 1"]).unwrap();
            let expected = vec![
                parsed.iter().nth(1).unwrap().clone(),
                parsed.iter().next().unwrap().clone(),
                parsed.iter().nth(1).unwrap().clone(),
            ];
            let formulas = expected.clone().into_iter().collect::<Formulas>();

            let mut borrowed = Vec::new();
            for formula in &formulas {
                borrowed.push(formula.clone());
            }
            assert_eq!(borrowed, expected);

            assert_eq!(formulas.len(), 3);
        }

        #[test]
        fn lookup_matches_formula_identity() {
            let formulas = parse("y ~ x + z - 1").unwrap();
            let first = formulas.iter().next().unwrap();

            assert!(std::ptr::eq(
                formulas.get_by_str(" y~z+x -1 ").unwrap().unwrap(),
                first,
            ));
            assert!(std::ptr::eq(
                formulas
                    .get_by_str("y ~ sw(x + z, x + z) - 1")
                    .unwrap()
                    .unwrap(),
                first,
            ));
            assert!(formulas.get_by_str("y ~ missing").unwrap().is_none());
            assert!(formulas.get_by_str("y ~").is_err());
            let error = formulas.get_by_str("y ~ sw(x, z)").unwrap_err();
            assert!(error.to_string().contains("exactly one formula"));
        }

        #[test]
        fn empty_collections_support_the_same_accessors() {
            let parsed = parse_many(std::iter::empty::<&str>()).unwrap();
            let collected = std::iter::empty::<Formula>().collect::<Formulas>();

            assert_eq!(collected, parsed);
            assert!(parsed.is_empty());
            assert_eq!(parsed.len(), 0);
            assert_eq!(parsed.iter().next(), None);
        }
    }

    fn fit(source: &str, data: &DataFrame) -> FittedFormula {
        let models = crate::parse(source).unwrap().fit(data).unwrap();
        assert_eq!(models.len(), 1);
        models.iter().next().unwrap().clone()
    }

    #[test]
    fn formula_tree_preserves_local_frames_through_fit_and_transform() {
        let mut root = crate::parse("y ~ x - 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let mut child = crate::parse("a ~ z - 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let grandchild = crate::parse("b ~ w - 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let sibling = crate::parse("c ~ v - 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        child
            .expressions
            .get_mut(&Role::Dependent)
            .unwrap()
            .children
            .push(grandchild);
        root.expressions
            .get_mut(&Role::Independent)
            .unwrap()
            .children = vec![child, sibling];

        let training = polars::df!(
            "y" => [1.0, 2.0], "x" => [3.0, 4.0],
            "a" => [5.0, 6.0], "z" => [7.0, 8.0],
            "b" => [9.0, 10.0], "w" => [11.0, 12.0],
            "c" => [13.0, 14.0], "v" => [15.0, 16.0],
        )
        .unwrap();
        let models = Formulas::new(vec![root.clone()]).fit(&training).unwrap();
        let model = models.iter().next().unwrap();
        assert_eq!(model.formula(), &root);

        let check = |model: &FittedFormula, data: &DataFrame| {
            let children = model.children(Role::Independent).unwrap();
            assert_eq!(children.len(), 2);
            let grandchildren = children[0].children(Role::Dependent).unwrap();
            assert_eq!(grandchildren.len(), 1);
            for (node, columns) in [
                (model, ["y", "x"]),
                (&children[0], ["a", "z"]),
                (&grandchildren[0], ["b", "w"]),
                (&children[1], ["c", "v"]),
            ] {
                assert!(
                    node.frame()
                        .unwrap()
                        .equals_missing(&data.select(columns).unwrap())
                );
            }
        };
        check(model, &training);

        let prediction = polars::df!(
            "y" => [20.0, 21.0, 22.0], "x" => [30.0, 31.0, 32.0],
            "a" => [40.0, 41.0, 42.0], "z" => [50.0, 51.0, 52.0],
            "b" => [60.0, 61.0, 62.0], "w" => [70.0, 71.0, 72.0],
            "c" => [80.0, 81.0, 82.0], "v" => [90.0, 91.0, 92.0],
        )
        .unwrap();
        for height in [0, 1, 3] {
            let input = prediction.head(Some(height));
            check(&model.transform(&input).unwrap(), &input);
            let transformed = models.transform(&input).unwrap();
            check(transformed.iter().next().unwrap(), &input);
        }
        check(model, &training);
    }

    #[test]
    fn model_select_transforms_only_selected_roles_and_their_children() {
        let data = polars::df!(
            "y" => [1.0, 2.0, 3.0], "g" => ["a", "b", "a"],
            "d" => [4.0, 5.0, 6.0], "i" => [7.0, 8.0, 9.0],
            "f" => [10.0, 11.0, 12.0],
        )
        .unwrap();
        for with_children in [false, true] {
            let mut formula = crate::parse("y ~ C(g) - 1")
                .unwrap()
                .iter()
                .next()
                .unwrap()
                .clone();
            let cases = [
                (Role::Dependent, Some("y"), "d"),
                (Role::Independent, Some("g"), "i"),
                (Role::FixedEffect, None, "f"),
            ];
            if with_children {
                for (role, _, column) in cases {
                    let child = crate::parse(&format!("{column} ~ 1"))
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .clone();
                    formula
                        .expressions
                        .get_mut(&role)
                        .unwrap()
                        .children
                        .push(child);
                }
            }
            let models = Formulas::new(vec![formula]).fit(&data).unwrap();
            let model = models.iter().next().unwrap();

            for (role, column, child_column) in cases {
                if !with_children && role == Role::FixedEffect {
                    assert!(model.expression(role).is_none());
                    continue;
                }
                let original = model.expression(role).unwrap();
                let training = original.frame().clone();
                let child_training = original
                    .children()
                    .first()
                    .map(|child| child.frame().unwrap());
                let selected = model.select(&[role]).unwrap();
                assert_eq!(selected.formula(), model.formula());
                assert!(selected.frame().unwrap().equals_missing(&training));
                let mut columns = column.into_iter().collect::<Vec<_>>();
                if with_children {
                    columns.push(child_column);
                }
                // No columns belonging to excluded roles or their children.
                let input = data.select(columns).unwrap().slice(1, 1);
                let applied = selected.transform(&input).unwrap();
                let applied_models = models.select(&[role]).unwrap().transform(&input).unwrap();

                for transformed in [&applied, applied_models.iter().next().unwrap()] {
                    assert_eq!(transformed.formula(), model.formula());
                    for (other, _, _) in cases {
                        if other != role {
                            assert!(selected.expression(other).is_none());
                            assert!(transformed.expression(other).is_none());
                            assert!(matches!(
                                selected.children(other),
                                Err(crate::SelectionError::MissingRole(missing)) if missing == other
                            ));
                            assert!(matches!(
                                transformed.children(other),
                                Err(crate::SelectionError::MissingRole(missing)) if missing == other
                            ));
                        }
                    }
                    let output = transformed.expression(role).unwrap();
                    assert!(output.frame().equals_missing(&training.slice(1, 1)));
                    assert_eq!(
                        transformed.children(role).unwrap().len(),
                        usize::from(with_children)
                    );
                    if let Some(child_training) = &child_training {
                        assert_eq!(selected.children(role).unwrap().len(), 1);
                        assert!(
                            transformed.children(role).unwrap()[0]
                                .frame()
                                .unwrap()
                                .equals_missing(&child_training.slice(1, 1))
                        );
                        assert!(
                            original.children()[0]
                                .frame()
                                .unwrap()
                                .equals_missing(child_training)
                        );
                    }
                }
                assert!(
                    model
                        .expression(role)
                        .unwrap()
                        .frame()
                        .equals_missing(&training)
                );
            }
        }
    }

    #[test]
    fn expression_addition_preserves_children_and_deduplicates_terms() {
        let first = crate::parse("a ~ z")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let second = crate::parse("b ~ w")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let term = Term::from(Factor::from("x"));
        let left = Expression::new([term.clone()], vec![first.clone()]);
        let right = Expression::new([term.clone()], vec![second.clone(), first.clone()]);

        let combined = left.add(right);

        assert_eq!(combined.terms, vec![term]);
        assert_eq!(combined.children, vec![first.clone(), second, first]);
    }

    #[test]
    fn fixed_effect_encoding_preserves_attached_children() {
        let child = crate::parse("a ~ z")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let expression = Expression::new(
            [Term::intercept(), Term::from(Factor::from("firm"))],
            vec![child.clone()],
        );

        let encoded = encode_fixed_effects(expression);
        let expected = crate::parse("y ~ x | firm")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .expressions[&Role::FixedEffect]
            .clone();

        assert_eq!(encoded.terms, expected.terms);
        assert_eq!(encoded.children, vec![child]);
    }

    #[test]
    fn factor_labels_round_trip_through_the_parser() {
        for source in [
            "log(x)",
            "log(`firm name`)",
            "log(`a``b`)",
            "log(`1`)",
            "add(x, 2)",
            "add(x, 2.0)",
            "C(`group name`, reference='a')",
        ] {
            let original = parse_expression.parse(source).unwrap();
            let label = original[0].terms[0].factors[0].render();
            assert_eq!(
                parse_expression.parse(&label).unwrap(),
                original,
                "{source}"
            );
        }
    }

    #[test]
    fn formula_display_round_trips_expanded_formulas() {
        for source in [
            "y ~ x",
            "y ~ x - 1",
            "y ~ x - 1 + 1",
            "y ~ x - 1 + 1 + z",
            "y ~ C(`group name`, reference='a') + poly(x, degree=2)",
            "`outcome value` ~ log(`1`) + add(x, 2.0)",
            "y ~ x | firm:year",
            "y ~ sw(x, z) | `firm name`",
            "y ~ x + [a ~ z]",
            "y ~ x - 1 + [a ~ z - 1]",
            "y ~ [a ~ 0] - 1",
            "0 ~ x + [a ~ z]",
            "[a ~ z] ~ x",
            "y ~ x + [a ~ z] + [b ~ w]",
            "y ~ x + [a ~ z + [b ~ w - 1]]",
            "y ~ x + [a ~ z | firm:year] | region",
            "y ~ sw(x, w) + [sw(a, b) ~ sw(z, v)]",
            "y ~ `a[b]` + [`child response` ~ log(`x[z]`) - 1]",
        ] {
            for formula in &crate::parse(source).unwrap() {
                let rendered = formula.to_string();
                let parsed = crate::parse(&rendered).unwrap();
                assert_eq!(
                    parsed,
                    std::iter::once(formula.clone()).collect::<Formulas>(),
                    "{rendered}"
                );
            }
        }
    }

    mod polynomial {
        use super::*;
        use crate::transform::TRANSFORM_CALLS;

        #[test]
        fn raw_fitting_and_prediction_preserve_values_and_schema() {
            let training = polars::df!(
                "y" => [1.0, 2.0],
                "x" => [2.0, -3.0],
            )
            .unwrap();
            let model = fit("y ~ poly(x, degree=3, raw=true)", &training);
            let expected = polars::df!(
                "Intercept" => [1.0; 2],
                "poly(x, degree=3, raw=true)[1]" => [2.0, -3.0],
                "poly(x, degree=3, raw=true)[2]" => [4.0, 9.0],
                "poly(x, degree=3, raw=true)[3]" => [8.0, -27.0],
            )
            .unwrap();
            assert!(
                model
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected)
            );

            let prediction = polars::df!("y" => [6.0, 7.0], "x" => [Some(4.0), None]).unwrap();
            let output = model.transform(&prediction).unwrap();
            let expected = polars::df!(
                "Intercept" => [1.0; 2],
                "poly(x, degree=3, raw=true)[1]" => [Some(4.0), None],
                "poly(x, degree=3, raw=true)[2]" => [Some(16.0), None],
                "poly(x, degree=3, raw=true)[3]" => [Some(64.0), None],
            )
            .unwrap();
            assert!(
                output
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected)
            );
        }

        #[test]
        fn orthogonal_prediction_reuses_coefficients_without_refitting() {
            let training = polars::df!(
                "y" => [1.0; 3],
                "x" => [-1.0, 0.0, 1.0],
            )
            .unwrap();
            let prediction = polars::df!("y" => [2.0, 3.0], "x" => [Some(2.0), None]).unwrap();
            for as_child in [false, true] {
                let formula = crate::parse("y ~ poly(x, degree=2)")
                    .unwrap()
                    .iter()
                    .next()
                    .unwrap()
                    .clone();
                let formula = if as_child {
                    let mut parent = crate::parse("y - y ~ 0")
                        .unwrap()
                        .iter()
                        .next()
                        .unwrap()
                        .clone();
                    parent
                        .expressions
                        .get_mut(&Role::Independent)
                        .unwrap()
                        .children
                        .push(formula);
                    parent
                } else {
                    formula
                };
                let models = Formulas::new(vec![formula]).fit(&training).unwrap();
                let model = models.iter().next().unwrap();
                let training_node = if as_child {
                    &model.children(Role::Independent).unwrap()[0]
                } else {
                    model
                };
                assert_eq!(
                    training_node
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .shape(),
                    (3, 3)
                );

                for use_collection in [false, true] {
                    TRANSFORM_CALLS.set((0, 0));
                    let output = if use_collection {
                        models
                            .transform(&prediction)
                            .unwrap()
                            .iter()
                            .next()
                            .unwrap()
                            .clone()
                    } else {
                        model.transform(&prediction).unwrap()
                    };
                    assert_eq!(TRANSFORM_CALLS.get(), (0, 1));
                    let node = if as_child {
                        &output.children(Role::Independent).unwrap()[0]
                    } else {
                        &output
                    };
                    let frame = node.expression(Role::Independent).unwrap().frame();
                    assert_eq!(
                        frame.get_column_names(),
                        [
                            "Intercept",
                            "poly(x, degree=2, raw=false)[1]",
                            "poly(x, degree=2, raw=false)[2]"
                        ]
                    );
                    for (column, expected) in frame.columns()[1..]
                        .iter()
                        .zip([2.0 / 2.0_f64.sqrt(), 10.0 / 6.0_f64.sqrt()])
                    {
                        assert!((column.f64().unwrap().get(0).unwrap() - expected).abs() < 1e-12);
                        assert_eq!(column.f64().unwrap().get(1), None);
                    }
                }
            }
        }

        #[test]
        fn empty_predictions_retain_both_basis_schemas() {
            let training = polars::df!("y" => [1.0; 3], "x" => [-1.0, 0.0, 1.0]).unwrap();
            let empty = polars::df!("y" => Vec::<f64>::new(), "x" => Vec::<f64>::new()).unwrap();
            for source in ["y ~ poly(x, degree=2)", "y ~ poly(x, degree=2, raw=true)"] {
                let model = fit(source, &training);
                let output = model.transform(&empty).unwrap();
                assert_eq!(
                    output
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .shape(),
                    (0, 3)
                );
                assert_eq!(
                    output.expression(Role::Dependent).unwrap().frame().shape(),
                    (0, 1)
                );
                assert_eq!(
                    output
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .schema(),
                    model
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .schema()
                );
            }
        }

        #[test]
        fn nested_numeric_inputs_match_explicit_transforms() {
            let data = polars::df!(
                "y" => [1.0; 3], "x" => [1.0, std::f64::consts::E, std::f64::consts::E.powi(2)],
            )
            .unwrap();
            let expected = fit("y ~ log(x) + mul(log(x), log(x))", &data);
            TRANSFORM_CALLS.set((0, 0));
            let actual = fit("y ~ poly(log(x), degree=2, raw=true)", &data);
            assert_eq!(TRANSFORM_CALLS.get(), (2, 2));
            assert_eq!(
                actual
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                [
                    "Intercept",
                    "poly(log(x), degree=2, raw=true)[1]",
                    "poly(log(x), degree=2, raw=true)[2]",
                ]
            );
            for (actual, expected) in actual
                .expression(Role::Independent)
                .unwrap()
                .frame()
                .columns()
                .iter()
                .zip(
                    expected
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .columns(),
                )
            {
                assert_eq!(
                    actual.f64().unwrap().iter().collect::<Vec<_>>(),
                    expected.f64().unwrap().iter().collect::<Vec<_>>()
                );
            }
        }

        #[test]
        fn basis_by_column_interactions_preserve_order_and_values() {
            let data =
                polars::df!("y" => [1.0, 2.0], "x" => [2.0, 3.0], "z" => [5.0, 7.0]).unwrap();
            let model = fit("y ~ poly(x, degree=2, raw=true):z - 1", &data);
            let expected = polars::df!(
                "poly(x, degree=2, raw=true)[1]:z" => [10.0, 21.0],
                "poly(x, degree=2, raw=true)[2]:z" => [20.0, 63.0],
            )
            .unwrap();
            assert!(
                model
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected)
            );
        }

        #[test]
        fn basis_by_basis_interactions_form_the_cartesian_product() {
            let data =
                polars::df!("y" => [1.0, 2.0], "x" => [2.0, 3.0], "z" => [5.0, 7.0]).unwrap();
            let model = fit(
                "y ~ poly(x, degree=2, raw=true):poly(z, degree=2, raw=true) - 1",
                &data,
            );
            let expected = polars::df!(
                "poly(x, degree=2, raw=true)[1]:poly(z, degree=2, raw=true)[1]" => [10.0, 21.0],
                "poly(x, degree=2, raw=true)[1]:poly(z, degree=2, raw=true)[2]" => [50.0, 147.0],
                "poly(x, degree=2, raw=true)[2]:poly(z, degree=2, raw=true)[1]" => [20.0, 63.0],
                "poly(x, degree=2, raw=true)[2]:poly(z, degree=2, raw=true)[2]" => [100.0, 441.0],
            )
            .unwrap();
            assert!(
                model
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected)
            );
        }

        #[test]
        fn polynomial_calls_reject_expanding_compositions() {
            let data = polars::df!("y" => [1.0], "x" => [2.0], "z" => [3.0], "g" => ["a"]).unwrap();
            for source in [
                "y ~ poly(C(g))",
                "y ~ log(poly(x))",
                "y ~ poly(poly(x))",
                "y ~ poly(add(x, poly(z)))",
            ] {
                assert!(
                    matches!(
                        crate::parse(source).unwrap().fit(&data),
                        Err(ModelError::Validation(
                            ValidationError::InvalidTransformComposition
                        ))
                    ),
                    "{source}"
                );
            }
        }
    }

    #[test]
    fn multiplicative_operators_preserve_column_order() {
        let data = polars::df!("a" => [2.0], "b" => [3.0], "c" => [4.0]).unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("a*b*c", &["a", "b", "a:b", "c", "a:c", "b:c", "a:b:c"]),
            ("a*b/c", &["a", "b", "a:b", "a:b:c"]),
            ("a/b*c", &["a", "a:b", "c", "a:c", "a:b:c"]),
            ("a*(b/c)", &["a", "b", "b:c", "a:b", "a:b:c"]),
        ];
        for &(source, names) in cases {
            let model = fit(&format!("a ~ {source} - 1"), &data);
            let output = model.expression(Role::Independent).unwrap().frame();
            assert_eq!(output.get_column_names(), names, "{source}");
        }
    }

    #[test]
    fn nesting_fits_like_explicit_interactions() {
        let training = polars::df!(
            "y" => [1.0, 2.0],
            "a" => [2.0, 3.0],
            "b" => [Some(4.0), None],
            "c" => [5.0, 7.0],
        )
        .unwrap();
        let input = polars::df!(
            "y" => [3.0], "a" => [4.0], "b" => [6.0], "c" => [8.0],
        )
        .unwrap();
        let nested = crate::parse("y ~ (a+b)/c").unwrap().fit(&training).unwrap();
        let expanded = crate::parse("y ~ a+b+a:b:c")
            .unwrap()
            .fit(&training)
            .unwrap();
        let transformed_nested = nested.transform(&input).unwrap();
        let transformed_expanded = expanded.transform(&input).unwrap();

        for (actual, expected, interactions) in [
            (&nested, &expanded, vec![Some(40.0), None]),
            (
                &transformed_nested,
                &transformed_expanded,
                vec![Some(192.0)],
            ),
        ] {
            let actual = actual.iter().next().unwrap();
            let expected = expected.iter().next().unwrap();
            assert!(
                actual
                    .expression(Role::Dependent)
                    .unwrap()
                    .frame()
                    .equals_missing(expected.expression(Role::Dependent).unwrap().frame())
            );
            assert!(
                actual
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(expected.expression(Role::Independent).unwrap().frame())
            );
            assert_eq!(
                actual
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                ["Intercept", "a", "b", "a:b:c"]
            );
            assert_eq!(
                actual
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .column("a:b:c")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                interactions
            );
        }
    }

    #[test]
    fn literal_factors_retain_scalar_storage_at_current_input_height() {
        let expressions = parse_expression.parse("mul(add(x, -2), 3.0)").unwrap();
        let expression = expressions.first().unwrap();
        let scheduled = schedule_factors(&[expression]);
        let training = polars::df!("x" => [1.0, 2.0]).unwrap();
        let fitted = fit_and_transform_factors(&scheduled, &training).unwrap();
        let fitted_factors = fitted
            .iter()
            .map(|entry| Arc::clone(&entry.factor))
            .collect::<Vec<_>>();

        for height in [0, 1, 4] {
            let input = polars::df!("x" => vec![4.0; height]).unwrap();
            let transformed = transform_factors(&fitted_factors, &input).unwrap();
            for (entries, expected_height) in [(&fitted, training.height()), (&transformed, height)]
            {
                let mut literal_count = 0;
                for entry in entries {
                    if let Factor::Literal(value) = entry.factor.as_ref() {
                        literal_count += 1;
                        let [Column::Scalar(column)] = entry.columns.frame().columns() else {
                            panic!("literal must retain scalar storage");
                        };
                        assert_eq!(column.scalar(), value);
                        assert_eq!(column.len(), expected_height);
                    }
                }
                assert_eq!(literal_count, 2);
            }
        }
    }

    #[test]
    fn quoted_columns_preserve_names_and_values_on_fitting_and_transformation() {
        let training = polars::df!(
            "household income" => [4.0, 9.0],
            "income-growth" => [2.0, 3.0],
            "a+b" => [5.0, 6.0],
            "a`b" => [7.0, 8.0],
            "" => [10.0, 11.0],
            "1" => [12.0, 13.0],
        )
        .unwrap();
        let input = training.slice(1, 1);

        for (source, expected) in [
            (
                "`household income`",
                polars::df!("household income" => [4.0, 9.0]).unwrap(),
            ),
            ("`a+b`", polars::df!("a+b" => [5.0, 6.0]).unwrap()),
            ("`a``b`", polars::df!("a`b" => [7.0, 8.0]).unwrap()),
            ("``", polars::df!("" => [10.0, 11.0]).unwrap()),
            ("`1`", polars::df!("1" => [12.0, 13.0]).unwrap()),
            (
                "sqrt(`household income`)",
                polars::df!("sqrt(household income)" => [2.0, 3.0]).unwrap(),
            ),
            (
                "`household income`:`income-growth`",
                polars::df!("household income:income-growth" => [8.0, 27.0]).unwrap(),
            ),
        ] {
            let fitted = fit(&format!("`household income` ~ {source} - 1"), &training);
            assert!(
                fitted
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected),
                "{source}"
            );

            let transformed = fitted.transform(&input).unwrap();
            assert!(
                transformed
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected.slice(1, 1)),
                "{source}"
            );
        }
    }

    #[test]
    fn bare_nonnumeric_columns_are_rejected_before_interaction_casting() {
        let data = polars::df!(
            "y" => [1.0, 2.0, 3.0],
            "x" => [2.0, 3.0, 4.0],
            "group" => ["a", "b", "a"],
            "numeric_text" => ["2", "3", "2"],
            "flag" => [true, false, true],
        )
        .unwrap();

        for (source, column) in [
            ("y ~ group", "group"),
            ("group ~ x", "group"),
            ("y ~ numeric_text:x", "numeric_text"),
            ("y ~ x:numeric_text", "numeric_text"),
            ("y ~ flag", "flag"),
            ("flag ~ x", "flag"),
            ("y ~ flag:x", "flag"),
            ("y ~ x:flag", "flag"),
        ] {
            let specification = crate::parse(source).unwrap();
            let Err(ModelError::Transform(TransformError::NonNumericInput { name, actual })) =
                specification.fit(&data)
            else {
                panic!("expected NonNumericInput for `{source}`");
            };

            assert_eq!(name.as_str(), column);
            assert_eq!(actual, data.column(column).unwrap().dtype().to_string());
        }
    }

    #[test]
    fn transform_errors_identify_the_directly_failing_call() {
        let data = polars::df!("y" => [1.0, 2.0], "x" => ["a", "b"]).unwrap();

        for (source, call) in [
            ("y ~ log(x)", "log(x)"),
            ("y ~ exp(log(x))", "log(x)"),
            ("y ~ add(1, log(x))", "log(x)"),
        ] {
            let error = crate::parse(source).unwrap().fit(&data).err().unwrap();
            assert_eq!(
                error.to_string(),
                format!(
                    "{call}: column `x`: expected a primitive numeric dtype, received {}",
                    DataType::String,
                ),
                "{source}",
            );
        }
    }

    #[test]
    fn categorical_fit_errors_include_reference_and_original_input_name() {
        let data = polars::df!("y" => [1.0, 2.0], "group name" => ["a", "b"]).unwrap();
        let error = crate::parse("y ~ C(`group name`, reference='missing')")
            .unwrap()
            .fit(&data)
            .err()
            .unwrap();

        assert_eq!(
            error.to_string(),
            "C(`group name`, reference='missing'): `reference` value is not among the fitted levels",
        );
    }

    #[test]
    fn fitted_integer_encoding_looks_up_columns_by_name() {
        let data = polars::df!(
            "y" => [1.0, 2.0],
            "a" => ["x", "y"],
            "b" => [1_i32, 2],
        )
        .unwrap();
        let formula = Formula {
            expressions: BTreeMap::from([
                (
                    Role::Dependent,
                    Expression::from(Term::from(Factor::from("y"))),
                ),
                (
                    Role::Independent,
                    Expression::from(Term::from(Factor::Call {
                        transform: Transform::F(Fit::Unfitted(IntegerEncodingDefinition)),
                        arguments: ["a", "b"]
                            .into_iter()
                            .map(|name| Arc::new(Factor::from(name)))
                            .collect(),
                    })),
                ),
            ]),
        };
        let models = Formulas::new(vec![formula]).fit(&data).unwrap();
        let reordered = data.select(["b", "y", "a"]).unwrap();
        let transformed = models.transform(&reordered).unwrap();
        assert!(
            models
                .iter()
                .next()
                .unwrap()
                .expression(Role::Independent)
                .unwrap()
                .frame()
                .equals_missing(
                    transformed
                        .iter()
                        .next()
                        .unwrap()
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                )
        );

        let missing = data.select(["y", "a"]).unwrap();
        let error = models.transform(&missing).err().unwrap();
        assert!(matches!(
            error,
            ModelError::Transform(TransformError::Polars(_))
        ));
        assert!(error.to_string().contains('b'));
    }

    #[test]
    fn categorical_encoding_accepts_string_and_boolean_inputs() {
        let training = polars::df!(
            "y" => [1.0, 2.0, 3.0],
            "group" => ["a", "b", "a"],
            "flag" => [true, false, true],
        )
        .unwrap();
        let input = polars::df!(
            "group" => ["b", "a", "unseen"],
            "flag" => [false, true, false],
        )
        .unwrap();

        for (source, expected) in [
            (
                "C(group)",
                polars::df!(
                    "C(group)[a]" => [Some(false), Some(true), None],
                    "C(group)[b]" => [Some(true), Some(false), None],
                )
                .unwrap(),
            ),
            (
                "C(flag)",
                polars::df!(
                    "C(flag)[True]" => [false, true, false],
                    "C(flag)[False]" => [true, false, true],
                )
                .unwrap(),
            ),
        ] {
            let fitted = fit(&format!("y ~ {source} - 1"), &training)
                .select(&[Role::Independent])
                .unwrap();
            let transformed = fitted.transform(&input).unwrap();

            assert!(
                transformed
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected),
                "{source}"
            );
        }
    }

    #[test]
    fn transformation_requires_exact_dtypes_for_bare_and_transformed_inputs() {
        let training = polars::df!("y" => [1.0, 2.0], "x" => [2_i32, 3]).unwrap();
        let changed = [
            polars::df!("y" => [1.0, 2.0], "x" => [2_i64, 3]).unwrap(),
            polars::df!("y" => [1.0, 2.0], "x" => [2.0_f64, 3.0]).unwrap(),
            polars::df!("y" => [1.0, 2.0], "x" => ["2", "3"]).unwrap(),
            polars::df!("y" => [1.0, 2.0], "x" => [true, false]).unwrap(),
        ];

        for (source, output_dtype) in [
            ("y ~ x - 1", DataType::Int32),
            ("y ~ log(x) - 1", DataType::Float64),
        ] {
            let models = crate::parse(source).unwrap().fit(&training).unwrap();
            let model = models.iter().next().unwrap();
            assert_eq!(
                model.expression(Role::Independent).unwrap().frame()[0].dtype(),
                &output_dtype
            );
            assert!(models.transform(&training).is_ok());

            for input in &changed {
                // Every transformation entry point must use the original fitted dtype.
                for result in [
                    models.transform(input).map(|_| ()),
                    model.transform(input).map(|_| ()),
                    model
                        .select(&[Role::Independent])
                        .unwrap()
                        .transform(input)
                        .map(|_| ()),
                ] {
                    let expected_message = format!(
                        "input `x`: expected {}, received {}",
                        DataType::Int32,
                        input.column("x").unwrap().dtype(),
                    );
                    // Check the public ModelError, including its FitError wrapper.
                    assert_eq!(result.as_ref().unwrap_err().to_string(), expected_message);
                    let Err(ModelError::Fit(crate::FitError::Transform(
                        error @ TransformError::InputTypeMismatch { .. },
                    ))) = result
                    else {
                        panic!("expected InputTypeMismatch for `{source}`");
                    };

                    assert_eq!(error.to_string(), expected_message);
                }
            }
        }
    }

    #[test]
    fn empty_batch_fits_without_models() {
        let data = polars::df!("y" => [1.0]).unwrap();
        let models = crate::parse_many(std::iter::empty())
            .unwrap()
            .fit(&data)
            .unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn argument_count_errors_identify_the_failing_call_before_transformation() {
        let data = DataFrame::default();
        for (source, expected) in [
            (
                "y ~ log(x) + add(x) + add(y, z)",
                "add(x): invalid argument count: expected at least 2, received 1",
            ),
            (
                "y ~ exp(log(x, x))",
                "log(x, x): invalid argument count: expected exactly 1, received 2",
            ),
            (
                "y ~ log(add(x, 2), x)",
                "log(add(x, 2), x): invalid argument count: expected exactly 1, received 2",
            ),
            (
                "y ~ C(x, x)",
                "C(x, x): invalid argument count: expected exactly 1, received 2",
            ),
            (
                "y ~ C(x, x, reference='a')",
                "C(x, x, reference='a'): invalid argument count: expected exactly 1, received 2",
            ),
            (
                "y ~ poly(x, 2)",
                "poly(x, 2, degree=1, raw=false): invalid argument count: expected exactly 1, received 2",
            ),
        ] {
            let error = crate::parse(source).unwrap().fit(&data).err().unwrap();
            assert!(matches!(
                error,
                ModelError::Validation(ValidationError::InvalidArgumentCount { .. })
            ));
            assert_eq!(error.to_string(), expected, "{source}");
        }
    }

    #[test]
    fn batch_fitting_validates_all_formulas_before_fitting() {
        use crate::transform::TRANSFORM_CALLS;

        let data = polars::df!("y" => [1.0], "x" => [2.0]).unwrap();
        for invalid_child in [false, true] {
            let mut specification = crate::parse("y ~ sw(log(x), sqrt(x, x))").unwrap();
            if invalid_child {
                let child = specification.formulas.pop().unwrap();
                specification.formulas[0]
                    .expressions
                    .get_mut(&Role::Independent)
                    .unwrap()
                    .children
                    .push(child);
            }
            TRANSFORM_CALLS.set((0, 0));
            assert!(matches!(
                specification.fit(&data),
                Err(ModelError::Validation(
                    ValidationError::InvalidArgumentCount { .. }
                ))
            ));
            assert_eq!(TRANSFORM_CALLS.get(), (0, 0));
        }
    }

    mod multiple_estimation {
        use super::*;

        const INTERACTION: &str = "sqrt(exp(sqrt(g))):x";

        fn fixture() -> (Formulas, DataFrame) {
            let data = polars::df!(
                "y1" => [1.0, 2.0, 3.0], "y2" => [4.0, 5.0, 6.0],
                "g" => [Some(0.0), Some(1.0), None], "x" => [2.0, 3.0, 4.0],
                "w" => [7.0, 8.0, 9.0], "v" => [10.0, 11.0, 12.0],
            )
            .unwrap();
            let specification = crate::parse(
                "sw(y1, y2) ~ sw0(w, v) + sw(sqrt(exp(sqrt(g))):x, x:sqrt(exp(sqrt(g))))",
            )
            .unwrap();
            (specification, data)
        }

        #[test]
        fn preserves_model_contents_and_order() {
            let (specification, data) = fixture();
            assert_eq!(specification.len(), 6);
            let collection = specification.fit(&data).unwrap();
            assert_eq!(collection.len(), 6);
            let expected_interaction = Column::new(
                INTERACTION.into(),
                [Some(2.0), Some(3.0 * std::f64::consts::E.sqrt()), None],
            );

            let cases: &[(&str, &str, &[&str])] = &[
                (
                    "y1 ~ sqrt(exp(sqrt(g))):x",
                    "y1",
                    &["Intercept", INTERACTION],
                ),
                (
                    "y1 ~ w + sqrt(exp(sqrt(g))):x",
                    "y1",
                    &["Intercept", "w", INTERACTION],
                ),
                (
                    "y1 ~ v + sqrt(exp(sqrt(g))):x",
                    "y1",
                    &["Intercept", "v", INTERACTION],
                ),
                (
                    "y2 ~ sqrt(exp(sqrt(g))):x",
                    "y2",
                    &["Intercept", INTERACTION],
                ),
                (
                    "y2 ~ w + sqrt(exp(sqrt(g))):x",
                    "y2",
                    &["Intercept", "w", INTERACTION],
                ),
                (
                    "y2 ~ v + sqrt(exp(sqrt(g))):x",
                    "y2",
                    &["Intercept", "v", INTERACTION],
                ),
            ];
            let mut expected_columns = data.clone();
            expected_columns
                .with_column(Column::new("Intercept".into(), [1.0; 3]))
                .unwrap();
            expected_columns.with_column(expected_interaction).unwrap();

            for (model, &(source, response, names)) in collection.iter().zip(cases) {
                let expected_formula = crate::parse(source).unwrap();
                assert_eq!(
                    model.formula(),
                    expected_formula.iter().next().unwrap(),
                    "{source}"
                );
                assert_eq!(
                    model
                        .expression(Role::Dependent)
                        .unwrap()
                        .frame()
                        .get_column_names(),
                    [response]
                );
                assert!(
                    model.expression(Role::Dependent).unwrap().frame()[0]
                        .equals_missing(data.column(response).unwrap())
                );
                let frame = model.expression(Role::Independent).unwrap().frame();
                assert_eq!(frame.get_column_names(), names);
                for name in names {
                    assert!(
                        frame
                            .column(name)
                            .unwrap()
                            .equals_missing(expected_columns.column(name).unwrap()),
                        "{source}: {name}"
                    );
                }
            }
        }

        #[test]
        fn transforms_nested_transforms_once_per_batch() {
            use crate::transform::TRANSFORM_CALLS;

            let data =
                polars::df!("y" => [1.0, 2.0], "g" => [0.0, 1.0], "x" => [3.0, 4.0]).unwrap();
            let specification = crate::parse("y ~ sqrt(exp(sqrt(g))) + sw0(x)").unwrap();
            assert_eq!(specification.len(), 2);
            TRANSFORM_CALLS.set((0, 0));
            let _models = specification.fit(&data).unwrap();
            // Both sqrt calls and exp each fit and transform once across all models.
            assert_eq!(TRANSFORM_CALLS.get(), (3, 3));
        }

        #[test]
        fn shares_column_buffers_across_models() {
            let data =
                polars::df!("y" => [1.0, 2.0], "x" => [3.0, 4.0], "z" => [5.0, 6.0]).unwrap();
            let specification = crate::parse("y ~ x:z + sw0(x)").unwrap();
            let collection = specification.fit(&data).unwrap();
            assert_eq!(collection.len(), 2);
            let mut models = collection.iter();
            let first = models.next().unwrap();
            let second = models.next().unwrap();
            let values_ptr = |column: &polars::prelude::Column| {
                column
                    .f64()
                    .unwrap()
                    .downcast_iter()
                    .next()
                    .unwrap()
                    .values()
                    .as_ptr()
            };
            assert_eq!(
                values_ptr(
                    first
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .column("x:z")
                        .unwrap()
                ),
                values_ptr(
                    second
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .column("x:z")
                        .unwrap()
                ),
            );
            assert_eq!(
                values_ptr(&first.expression(Role::Dependent).unwrap().frame()[0]),
                values_ptr(&second.expression(Role::Dependent).unwrap().frame()[0])
            );
            assert_eq!(
                values_ptr(
                    second
                        .expression(Role::Independent)
                        .unwrap()
                        .frame()
                        .column("x")
                        .unwrap()
                ),
                values_ptr(data.column("x").unwrap())
            );
        }

        #[test]
        fn transform_runs_once_and_shares_all_columns() {
            use crate::transform::TRANSFORM_CALLS;

            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0, 4.0],
                "g" => [Some("c"), Some("a"), Some("b"), None],
                "x" => [Some(5.0), Some(6.0), Some(7.0), None],
                "z" => [2.0; 4],
            )
            .unwrap();
            let input = polars::df!(
                "y" => [4.0, 3.0, 2.0, 1.0],
                "g" => [Some("b"), Some("c"), None, Some("a")],
                "x" => [Some(8.0), None, Some(9.0), Some(10.0)],
                "z" => [3.0; 4],
            )
            .unwrap();

            for (source, width) in [
                ("y ~ C(g) + sw0(x)", 4),
                ("y ~ C(g, reference='a') + sw0(x)", 3),
                (
                    "y ~ (C(g, reference='a') + C(g, reference='a'))*z + sw0(x)",
                    6,
                ),
                (
                    "y ~ (poly(x, degree=2, raw=true) + poly(x, raw=true, degree=2))*z + sw0(x)",
                    6,
                ),
                (
                    "y ~ (poly(x, degree=2) + poly(x, raw=false, degree=2))*z + sw0(x)",
                    6,
                ),
                ("y ~ (poly(x) + poly(x, degree=1, raw=false))*z + sw0(x)", 4),
            ] {
                let specification = crate::parse(source).unwrap();
                TRANSFORM_CALLS.set((0, 0));
                let collection = specification.fit(&data).unwrap();
                assert_eq!(TRANSFORM_CALLS.get(), (1, 1), "{source}");

                TRANSFORM_CALLS.set((0, 0));
                let applied = collection.transform(&input).unwrap();
                assert_eq!(TRANSFORM_CALLS.get(), (0, 1), "{source}");

                for models in [&collection, &applied] {
                    assert_eq!(models.len(), 2);
                    let mut models = models
                        .iter()
                        .map(|model| model.expression(Role::Independent).unwrap().frame());
                    let first = models.next().unwrap();
                    let second = models.next().unwrap();
                    assert_eq!(first.shape(), (4, width), "{source}");
                    assert_eq!(second.shape(), (4, width + 1), "{source}");
                    let names = first.get_column_names();
                    assert_eq!(&second.get_column_names()[..width], names.as_slice());
                    assert_eq!(second.columns().last().unwrap().name().as_str(), "x");

                    for left in &first.columns()[1..] {
                        let right = second.column(left.name().as_str()).unwrap();
                        assert!(left.equals_missing(right), "{source}: {}", left.name());
                        // Sharing the Series shares both its values and validity buffers.
                        assert!(
                            Arc::ptr_eq(
                                &left.as_materialized_series().0,
                                &right.as_materialized_series().0,
                            ),
                            "{source}: {}",
                            left.name()
                        );
                    }
                }
            }
        }

        #[test]
        fn selected_model_outlives_specification_and_models() {
            let data = polars::df!(
                "y" => [1.0, 2.0, 3.0],
                "g" => [Some("a"), Some("b"), None],
                "x" => [2.0, 3.0, 4.0],
            )
            .unwrap();
            let specification = crate::parse("y ~ C(g) + sw0(x)").unwrap();
            let collection = specification.fit(&data).unwrap();
            let retained = (&collection)
                .into_iter()
                .next()
                .unwrap()
                .select(&[Role::Independent])
                .unwrap();
            drop(specification);
            drop(collection);
            drop(data);

            let expected = polars::df!(
                "Intercept" => [1.0; 3],
                "C(g)[a]" => [Some(true), Some(false), None],
                "C(g)[b]" => [Some(false), Some(true), None],
            )
            .unwrap();
            assert!(
                retained
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected)
            );

            let fresh = polars::df!(
                "g" => [Some("b"), Some("a"), Some("unseen"), None],
            )
            .unwrap();
            let transformed = retained.transform(&fresh).unwrap();
            let expected = polars::df!(
                "Intercept" => [1.0; 4],
                "C(g)[a]" => [Some(false), Some(true), None, None],
                "C(g)[b]" => [Some(true), Some(false), None, None],
            )
            .unwrap();
            assert!(
                transformed
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .equals_missing(&expected)
            );
        }
    }

    #[test]
    fn models_share_equivalent_interaction_buffers() {
        use crate::transform::TRANSFORM_CALLS;

        let specification = crate::parse_many([
            "g ~ sqrt(exp(sqrt(g))):x + w - 1",
            "g ~ x:sqrt(exp(sqrt(g))) + v - 1",
        ])
        .unwrap();
        let training = polars::df!(
            "g" => [0.0, 1.0], "x" => [2.0, 3.0],
            "w" => [6.0, 7.0], "v" => [8.0, 9.0],
        )
        .unwrap();
        TRANSFORM_CALLS.set((0, 0));
        let models = specification.fit(&training).unwrap();
        assert_eq!(TRANSFORM_CALLS.get(), (3, 3));

        let assert_shared = |models: &crate::FittedFormulas, expected: &[Option<f64>]| {
            let designs = models
                .iter()
                .map(|model| model.expression(Role::Independent).unwrap())
                .collect::<Vec<_>>();
            let name = "sqrt(exp(sqrt(g))):x";
            assert_eq!(designs[0].frame().get_column_names(), [name, "w"]);
            assert_eq!(designs[1].frame().get_column_names(), [name, "v"]);
            let left = designs[0].frame()[0].f64().unwrap();
            let right = designs[1].frame()[0].f64().unwrap();
            assert_eq!(left.len(), expected.len());
            assert!(designs[0].frame()[0].equals_missing(&designs[1].frame()[0]));
            for (actual, expected) in left.iter().zip(expected) {
                match (actual, expected) {
                    (Some(actual), Some(expected)) => {
                        assert!((actual - expected).abs() < 1e-12);
                    }
                    (None, None) => {}
                    values => panic!("unexpected interaction values: {values:?}"),
                }
            }
            // Inspect the values buffer directly: cont_slice rejects nulls.
            let left = left.downcast_iter().next().unwrap().values();
            let right = right.downcast_iter().next().unwrap().values();
            assert_eq!(left.as_ptr(), right.as_ptr());
        };
        assert_shared(
            &models,
            &[Some(2.0), Some(3.0 * std::f64::consts::E.sqrt())],
        );

        let input = polars::df!(
            "g" => [Some(1.0), Some(0.0), None], "x" => [4.0, 5.0, 6.0],
            "w" => [7.0, 8.0, 9.0], "v" => [10.0, 11.0, 12.0],
        )
        .unwrap();
        TRANSFORM_CALLS.set((0, 0));
        let applied = models.transform(&input).unwrap();
        assert_eq!(TRANSFORM_CALLS.get(), (0, 3));
        assert_shared(
            &applied,
            &[Some(4.0 * std::f64::consts::E.sqrt()), Some(5.0), None],
        );
    }

    #[test]
    fn shared_factors_execute_once_and_retain_shared_nodes() {
        use crate::transform::TRANSFORM_CALLS;

        let training = polars::df!("g" => [0.0, 1.0, 0.0], "x" => [2.0, 3.0, 4.0]).unwrap();
        TRANSFORM_CALLS.set((0, 0));
        let mut formula = crate::parse("sqrt(g) ~ sqrt(g):x + exp(sqrt(g)) - 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let child = crate::parse("sqrt(g) ~ exp(sqrt(g)) - 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        formula
            .expressions
            .get_mut(&Role::Independent)
            .unwrap()
            .children
            .push(child);
        let models = Formulas::new(vec![formula]).fit(&training).unwrap();
        let model = models.iter().next().unwrap();
        // Each transform runs once across both local roles and the child.
        assert_eq!(TRANSFORM_CALLS.get(), (2, 2));

        let sqrt = &model
            .expression(Role::Dependent)
            .unwrap()
            .expression()
            .terms[0]
            .factors[0];
        let independent = model.expression(Role::Independent).unwrap().expression();
        assert!(Arc::ptr_eq(sqrt, &independent.terms[0].factors[0]));
        let Factor::Call { arguments, .. } = independent.terms[1].factors[0].as_ref() else {
            panic!("expected exp call");
        };
        assert!(Arc::ptr_eq(sqrt, &arguments[0]));
        let child = &model.expression(Role::Independent).unwrap().children()[0];
        assert!(Arc::ptr_eq(
            sqrt,
            &child
                .expression(Role::Dependent)
                .unwrap()
                .expression()
                .terms[0]
                .factors[0]
        ));
        assert!(Arc::ptr_eq(
            &independent.terms[1].factors[0],
            &child
                .expression(Role::Independent)
                .unwrap()
                .expression()
                .terms[0]
                .factors[0]
        ));

        let input =
            polars::df!("g" => [Some(1.0), Some(0.0), None], "x" => [5.0, 6.0, 7.0]).unwrap();
        TRANSFORM_CALLS.set((0, 0));
        let applied = model.transform(&input).unwrap();
        assert_eq!(TRANSFORM_CALLS.get(), (0, 2));
        assert!(Arc::ptr_eq(
            sqrt,
            &applied
                .expression(Role::Dependent)
                .unwrap()
                .expression()
                .terms[0]
                .factors[0],
        ));
        assert!(Arc::ptr_eq(
            &independent.terms[1].factors[0],
            &applied
                .expression(Role::Independent)
                .unwrap()
                .expression()
                .terms[1]
                .factors[0],
        ));
        assert!(Arc::ptr_eq(
            sqrt,
            &applied
                .expression(Role::Independent)
                .unwrap()
                .expression()
                .terms[0]
                .factors[0],
        ));

        let expected_dependent = polars::df!("sqrt(g)" => [Some(1.0), Some(0.0), None]).unwrap();
        let expected_independent = polars::df!(
            "sqrt(g):x" => [Some(5.0), Some(0.0), None],
            "exp(sqrt(g))" => [Some(std::f64::consts::E), Some(1.0), None],
        )
        .unwrap();
        assert!(
            applied
                .expression(Role::Dependent)
                .unwrap()
                .frame()
                .equals_missing(&expected_dependent)
        );
        assert!(
            applied
                .expression(Role::Independent)
                .unwrap()
                .frame()
                .equals_missing(&expected_independent)
        );
        let child = &applied.expression(Role::Independent).unwrap().children()[0];
        assert!(
            child
                .expression(Role::Dependent)
                .unwrap()
                .frame()
                .equals_missing(&expected_dependent)
        );
        assert!(
            child
                .expression(Role::Independent)
                .unwrap()
                .frame()
                .equals_missing(&expected_independent.select(["exp(sqrt(g))"]).unwrap())
        );
    }

    #[test]
    fn separate_fittings_learn_independent_state() {
        let specification = crate::parse("y ~ C(g) - 1").unwrap();
        let first_training = polars::df!("y" => [1.0, 2.0], "g" => ["a", "b"]).unwrap();
        let second_training = polars::df!("y" => [1.0, 2.0], "g" => ["b", "c"]).unwrap();
        let first = specification
            .fit(&first_training)
            .unwrap()
            .select(&[Role::Independent])
            .unwrap();
        let second = specification
            .fit(&second_training)
            .unwrap()
            .select(&[Role::Independent])
            .unwrap();

        let input = polars::df!("g" => ["a", "b", "c"]).unwrap();
        let expected_first = polars::df!(
            "C(g)[a]" => [Some(true), Some(false), None],
            "C(g)[b]" => [Some(false), Some(true), None],
        )
        .unwrap();
        let expected_second = polars::df!(
            "C(g)[b]" => [None, Some(true), Some(false)],
            "C(g)[c]" => [None, Some(false), Some(true)],
        )
        .unwrap();
        for (models, expected) in [(&first, &expected_first), (&second, &expected_second)] {
            let transformed = models.transform(&input).unwrap();
            let frame = transformed
                .iter()
                .next()
                .unwrap()
                .expression(Role::Independent)
                .unwrap()
                .frame();
            assert!(frame.equals_missing(expected));
        }
    }

    #[test]
    fn public_pipeline_types_remain_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<Formulas>();
        assert_send_sync::<Formula>();
        assert_send_sync::<crate::FittedFormulas>();
        assert_send_sync::<FittedFormula>();
    }

    #[test]
    fn planning_deduplicates_factors_across_formula_sides() {
        for (source, count, right_index) in [
            ("log(x) ~ log(x) - 1", 2, 0),
            ("C(x) ~ C(x) - 1", 2, 0),
            ("C(x, reference='a') ~ C(x, reference='a') - 1", 2, 0),
            ("C(x):z ~ z:C(x) - 1", 3, 1),
            (
                "C(x, reference='a'):C(z) ~ C(z):C(x, reference='a') - 1",
                4,
                1,
            ),
            ("poly(x, raw=true) ~ poly(x, degree=1, raw=true) - 1", 2, 0),
            ("poly(x) ~ poly(x, degree=1, raw=false) - 1", 2, 0),
            ("poly(x, degree=2) ~ poly(x, raw=false, degree=2) - 1", 2, 0),
        ] {
            let formula = crate::parse(source).unwrap().iter().next().unwrap().clone();
            let scheduled = schedule_factors(&[
                &formula.expressions[&Role::Dependent],
                &formula.expressions[&Role::Independent],
            ]);

            assert_eq!(scheduled.len(), count, "{source}");
            assert_eq!(scheduled[0].as_ref(), &Factor::from("x"));
            assert_eq!(
                formula.expressions[&Role::Dependent].terms[0].factors[0],
                formula.expressions[&Role::Independent].terms[0].factors[right_index],
                "{source}"
            );
            for factor in formula.expressions[&Role::Dependent].terms[0]
                .factors
                .iter()
                .chain(&formula.expressions[&Role::Independent].terms[0].factors)
            {
                assert_eq!(
                    scheduled.iter().filter(|entry| *entry == factor).count(),
                    1,
                    "{source}: {}",
                    factor.render()
                );
            }
        }
    }

    #[test]
    fn planning_orders_dependencies_before_parents() {
        let expressions = parse_expression.parse("log(x) + sqrt(log(x))").unwrap();
        let expression = expressions.first().unwrap();
        let scheduled = schedule_factors(&[expression]);

        assert_eq!(scheduled.len(), 3);
        assert_eq!(scheduled[0].as_ref(), &Factor::from("x"));
        assert_eq!(scheduled[1], expression.terms[0].factors[0]);
        assert_eq!(scheduled[2], expression.terms[1].factors[0]);
    }

    #[test]
    fn planning_preserves_repeated_arguments() {
        let expressions = parse_expression.parse("sub(x, x)").unwrap();
        let expression = expressions.first().unwrap();
        let scheduled = schedule_factors(&[expression]);

        assert_eq!(scheduled.len(), 2);
        let Factor::Call { arguments, .. } = scheduled[1].as_ref() else {
            panic!("expected sub call");
        };

        assert_eq!(arguments.len(), 2);
        assert_eq!(arguments[0], scheduled[0]);
        assert_eq!(arguments[1], scheduled[0]);
    }

    #[test]
    fn planning_distinguishes_argument_order() {
        let expressions = parse_expression.parse("sub(x, y) + sub(y, x)").unwrap();
        let expression = expressions.first().unwrap();
        let scheduled = schedule_factors(&[expression]);

        assert_eq!(scheduled.len(), 4);
        assert_eq!(scheduled[0].as_ref(), &Factor::from("x"));
        assert_eq!(scheduled[1].as_ref(), &Factor::from("y"));
        assert_eq!(scheduled[2], expression.terms[0].factors[0]);
        assert_eq!(scheduled[3], expression.terms[1].factors[0]);
        assert_ne!(scheduled[2], scheduled[3]);
    }

    #[test]
    fn planning_distinguishes_transform_parameters() {
        for (source, input) in [
            ("C(g, reference='a') + C(g, reference='b')", "g"),
            ("poly(x) + poly(x, degree=2)", "x"),
            ("poly(x, raw=true) + poly(x, degree=2, raw=true)", "x"),
            (
                "poly(x, degree=2, raw=false) + poly(x, degree=2, raw=true)",
                "x",
            ),
        ] {
            let expressions = parse_expression.parse(source).unwrap();
            let expression = expressions.first().unwrap();
            let scheduled = schedule_factors(&[expression]);

            assert_eq!(scheduled.len(), 3, "{source}");
            assert_eq!(scheduled[0].as_ref(), &Factor::from(input));
            assert_eq!(scheduled[1], expression.terms[0].factors[0]);
            assert_eq!(scheduled[2], expression.terms[1].factors[0]);
            assert_ne!(scheduled[1], scheduled[2]);
        }
    }

    #[test]
    fn planning_excludes_cancelled_factors_and_intercepts() {
        let formula = crate::parse("y ~ sqrt(x) - sqrt(x) + 1")
            .unwrap()
            .iter()
            .next()
            .unwrap()
            .clone();
        let scheduled = schedule_factors(&[
            &formula.expressions[&Role::Dependent],
            &formula.expressions[&Role::Independent],
        ]);

        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].as_ref(), &Factor::from("y"));
        assert!(schedule_factors(&[&formula.expressions[&Role::Independent]]).is_empty());
    }

    #[test]
    fn intercept_is_first_and_included_once() {
        let data = polars::df!("y" => [2.0, 3.0], "x" => [4.0, 5.0]).unwrap();
        for source in ["y ~ x", "y ~ x + 1", "y ~ 1 + x + 1"] {
            let model = fit(source, &data);
            let frame = model.expression(Role::Independent).unwrap().frame();
            assert_eq!(frame.get_column_names(), ["Intercept", "x"]);
            assert_eq!(
                frame[0].f64().unwrap().iter().collect::<Vec<_>>(),
                vec![Some(1.0), Some(1.0)]
            );
            assert_eq!(
                model
                    .expression(Role::Dependent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                ["y"]
            );
        }
    }

    #[test]
    fn intercept_addition_and_subtraction_follow_term_order() {
        let data = polars::df!("y" => [2.0], "x" => [3.0]).unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("y ~ -1 + x", &["x"]),
            ("y ~ - 1 + x", &["x"]),
            ("y ~ x - 1", &["x"]),
            ("y ~ x - 1 + 1", &["x", "Intercept"]),
            ("y ~ x - 1 + 1 - 1", &["x"]),
            ("y ~ 0", &[]),
            ("y ~ 0 + x", &["x"]),
            ("y ~ x + 0", &["x"]),
            ("y ~ x + 0 + 1", &["x", "Intercept"]),
            ("y ~ x - 0", &["Intercept", "x"]),
            ("y ~ x - 1 - 0", &["x", "Intercept"]),
            ("y ~ -0 + x", &["Intercept", "x"]),
        ];
        for &(source, expected) in cases {
            let model = fit(source, &data);
            let independent = model.expression(Role::Independent);

            if expected.is_empty() {
                assert!(independent.is_none(), "{source}");
            } else {
                assert_eq!(
                    independent.unwrap().frame().get_column_names(),
                    expected,
                    "{source}",
                );
            }
        }
    }

    #[test]
    fn grouped_subtraction_keeps_intercept_removal_local() {
        let data = polars::df!("y" => [2.0], "x" => [3.0], "z" => [4.0]).unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("y ~ (x - 1)", &["Intercept", "x"]),
            ("y ~ (1 + x - 1)", &["Intercept", "x"]),
            ("y ~ (-1 + x)", &["Intercept", "x"]),
            ("y ~ (0 + x)", &["Intercept", "x"]),
            ("y ~ x - (z + 1)", &["x"]),
        ];
        for &(source, expected) in cases {
            let model = fit(source, &data);
            assert_eq!(
                model
                    .expression(Role::Independent)
                    .unwrap()
                    .frame()
                    .get_column_names(),
                expected,
                "{source}"
            );
        }
    }

    #[test]
    fn intercept_name_collision_requires_suppression() {
        let data = polars::df!("y" => [2.0], "Intercept" => [3.0]).unwrap();
        assert!(matches!(
            crate::parse("y ~ Intercept").unwrap().fit(&data),
            Err(ModelError::Transform(TransformError::Polars(
                polars::error::PolarsError::Duplicate(_)
            )))
        ));
        let model = fit("y ~ Intercept - 1", &data);
        assert_eq!(
            model.expression(Role::Independent).unwrap().frame(),
            &data.select(["Intercept"]).unwrap()
        );
    }

    #[test]
    fn intercept_uses_new_input_height_on_transformation() {
        let training = polars::df!("y" => [2.0, 3.0]).unwrap();
        let model = fit("y ~ 1", &training);
        for values in [vec![4.0], vec![4.0, 5.0, 6.0], vec![]] {
            let height = values.len();
            let input = polars::df!("y" => values).unwrap();
            let output = model.transform(&input).unwrap();
            let frame = output.expression(Role::Independent).unwrap().frame();
            assert_eq!(frame.shape(), (height, 1));
            assert_eq!(
                frame[0].f64().unwrap().iter().collect::<Vec<_>>(),
                vec![Some(1.0); height]
            );
            let Column::Scalar(column) = &frame[0] else {
                panic!("intercept must retain scalar storage");
            };
            assert_eq!(column.scalar(), &Scalar::from(1.0_f64));
        }
    }

    #[test]
    fn cancelled_terms_preserve_model_row_counts() {
        let specification = crate::parse("y ~ x-x-1").unwrap();
        for values in [vec![1.0, 2.0, 3.0], vec![]] {
            let height = values.len();
            let data = polars::df!("y" => values.clone(), "x" => values).unwrap();
            let models = specification.fit(&data).unwrap();
            let model = models.iter().next().unwrap();
            assert_eq!(
                model.expression(Role::Dependent).unwrap().frame().shape(),
                (height, 1)
            );
            assert!(model.expression(Role::Independent).is_none());
            let input = polars::df!("y" => [4.0, 5.0]).unwrap();
            let transformed = model.transform(&input).unwrap();
            assert_eq!(
                transformed
                    .expression(Role::Dependent)
                    .unwrap()
                    .frame()
                    .height(),
                2
            );
            assert!(transformed.expression(Role::Independent).is_none());
        }
    }

    #[test]
    fn interaction_identity_preserves_function_argument_order() {
        let expressions = parse_expression.parse("sub(a,b):x + x:sub(b,a)").unwrap();
        let expression = expressions.first().unwrap();
        assert_eq!(expression.terms.len(), 2);
    }

    #[test]
    fn interaction_deduplication_preserves_first_output_name() {
        let data = polars::df!("a" => [2.0, 4.0], "b" => [3.0, 5.0]).unwrap();
        let model = fit("a ~ b:a + a:b - 1", &data);
        let output = model.expression(Role::Independent).unwrap().frame();
        assert_eq!(output.shape(), (2, 1));
        assert_eq!(output.columns()[0].name().as_str(), "b:a");
    }

    fn assert_requires_one_argument(make_transform: impl Fn() -> Transform) {
        for count in [0, 1, 2, 3] {
            let arguments = (0..count)
                .map(|index| Arc::new(Factor::from(format!("x{index}").as_str())))
                .collect();

            let factor = Factor::Call {
                transform: make_transform(),
                arguments,
            };

            let result = factor.validate();

            if count == 1 {
                assert!(
                    result.is_ok(),
                    "{factor:?}: expected success, got {result:?}",
                );
            } else {
                assert!(
                    matches!(
                        result,
                        Err(ValidationError::InvalidArgumentCount {
                            expected: ArgumentCount::Exactly(1),
                            actual,
                            ..
                        }) if actual == count
                    ),
                    "{factor:?}: expected an argument-count error, got {result:?}",
                );
            }
        }
    }

    #[test]
    fn log_requires_exactly_one_argument() {
        assert_requires_one_argument(|| Transform::Expression(ExpressionTransform::LOG));
    }

    #[test]
    fn exp_requires_exactly_one_argument() {
        assert_requires_one_argument(|| Transform::Expression(ExpressionTransform::EXP));
    }

    #[test]
    fn polynomial_requires_exactly_one_argument() {
        assert_requires_one_argument(|| {
            Transform::Poly(Fit::Unfitted(
                PolynomialDefinition::new(None, None).unwrap(),
            ))
        });
    }

    #[test]
    fn categorical_requires_exactly_one_argument() {
        assert_requires_one_argument(|| {
            Transform::C(Fit::Unfitted(CategoricalDefinition::new(None)))
        });

        assert_requires_one_argument(|| {
            Transform::C(Fit::Unfitted(CategoricalDefinition::new(Some(
                Scalar::from(1_i64),
            ))))
        });
    }

    #[test]
    fn integer_encoding_requires_at_least_one_argument_and_allows_repetition() {
        for count in [0, 1, 2, 3, 16] {
            let factor = Factor::Call {
                transform: Transform::F(Fit::Unfitted(IntegerEncodingDefinition)),
                arguments: (0..count).map(|_| Arc::new(Factor::from("x"))).collect(),
            };
            let result = factor.validate();
            if count == 0 {
                assert!(matches!(
                    result,
                    Err(ValidationError::InvalidArgumentCount {
                        call,
                        expected: ArgumentCount::AtLeast(1),
                        actual: 0,
                    }) if call == "F()"
                ));
            } else {
                assert!(result.is_ok(), "{count} arguments: {result:?}");
            }
        }
    }

    #[test]
    fn integer_encoding_rejects_expanding_arguments_in_any_position() {
        for position in 0..3 {
            let mut arguments: Vec<_> = (0..3).map(|_| Arc::new(Factor::from("x"))).collect();
            arguments[position] = Arc::new(Factor::Call {
                transform: Transform::C(Fit::Unfitted(CategoricalDefinition::new(None))),
                arguments: vec![Arc::new(Factor::from("group"))],
            });
            let factor = Factor::Call {
                transform: Transform::F(Fit::Unfitted(IntegerEncodingDefinition)),
                arguments,
            };
            assert!(matches!(
                factor.validate(),
                Err(ValidationError::InvalidTransformComposition)
            ));
        }
    }

    #[test]
    fn model_transformation_reuses_fitted_integer_encoding() {
        let training = polars::df!(
            "y" => [1.0, 2.0, 3.0],
            "f1" => ["a", "b", "a"],
            "f2" => [1_i32, 1, 2],
        )
        .unwrap();
        let input = polars::df!(
            "y" => [4.0, 5.0, 6.0, 7.0],
            "f1" => ["a", "b", "a", "unseen"],
            "f2" => [2_i32, 1, 1, 1],
        )
        .unwrap();
        let model = fit("y ~ -1 | f1:f2", &training);
        let transformed = model.transform(&input).unwrap();
        let output = transformed.expression(Role::FixedEffect).unwrap().frame();

        assert_eq!(output.width(), 1);
        assert_eq!(output[0].name().as_str(), "f1:f2");
        assert_eq!(output[0].dtype(), &DataType::UInt32);
        assert_eq!(
            output[0].u32().unwrap().iter().collect::<Vec<_>>(),
            vec![Some(2), Some(1), Some(0), None]
        );
        assert_eq!(
            transformed.expression(Role::Dependent).unwrap().frame()[0].len(),
            4
        );

        let original = model.expression(Role::FixedEffect).unwrap().frame()[0]
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();
        assert_eq!(original, vec![Some(0), Some(1), Some(2)]);
        assert_eq!(
            model.expression(Role::FixedEffect).unwrap().expression(),
            transformed
                .expression(Role::FixedEffect)
                .unwrap()
                .expression()
        );
    }

    #[test]
    fn nested_fitting_matches_explicit_transform_composition() {
        let training = polars::df!(
            "y" => [1.0, 2.0, 3.0],
            "f1" => [0.0, 1.0, 0.0],
            "f2" => [1.0, 1.0, 2.0],
        )
        .unwrap();
        let input = polars::df!(
            "y" => [4.0, 5.0, 6.0, 7.0],
            "f1" => [0.0, 1.0, 2.0, 1.0],
            "f2" => [2.0, 1.0, 1.0, 2.0],
        )
        .unwrap();
        let model = fit("y ~ exp(add(f1, f2)) - 1", &training);
        let transformed = model.transform(&input).unwrap();

        let inner = ExpressionTransform::ADD.fit();
        let outer = ExpressionTransform::EXP.fit();

        for (data, design) in [
            (&training, model.expression(Role::Independent).unwrap()),
            (&input, transformed.expression(Role::Independent).unwrap()),
        ] {
            let inputs = [
                data.column("f1").unwrap().clone(),
                data.column("f2").unwrap().clone(),
            ];
            let summed = inner.transform(&inputs).unwrap();
            let expected = outer.transform(&[summed.frame()[0].clone()]).unwrap();
            assert!(design.frame().equals_missing(expected.frame()));
        }
    }
}
