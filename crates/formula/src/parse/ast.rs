use crate::Factor;
use std::sync::Arc;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum ConstructionError {
    Unsupported,
}

impl std::fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported => f.write_str("operation is not supported"),
        }
    }
}

impl std::error::Error for ConstructionError {}

pub(super) struct Permitted;
pub(super) enum Forbidden {}

pub(super) trait Permission: Sized {
    fn permit() -> Result<Self, ConstructionError>;
}

impl Permission for Permitted {
    fn permit() -> Result<Self, ConstructionError> {
        Ok(Self)
    }
}

impl Permission for Forbidden {
    fn permit() -> Result<Self, ConstructionError> {
        Err(ConstructionError::Unsupported)
    }
}

impl<E: ExpressionKind> TryFrom<ExpressionAst<E>> for Forbidden {
    type Error = ConstructionError;

    fn try_from(_: ExpressionAst<E>) -> Result<Self, Self::Error> {
        Err(ConstructionError::Unsupported)
    }
}

/// Check if this constructor is permitted
pub(super) fn permitted<P: Permission, T, E>(
    value: T,
    constructor: impl FnOnce(P, T) -> E,
) -> Result<E, ConstructionError> {
    Ok(constructor(P::permit()?, value))
}

/// Two operands of the same expression type
pub(super) struct Binary<E> {
    pub(super) left: Box<E>,
    pub(super) right: Box<E>,
}

impl<E> Binary<E> {
    pub(super) fn new(left: E, right: E) -> Self {
        Self {
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

pub(super) enum BinaryOperator {
    Add,
    Subtract,
    Interact,
    InteractFully,
    Nest,
}

impl BinaryOperator {
    fn construct<K: ExpressionKind>(
        self,
        operands: Binary<ExpressionAst<K>>,
    ) -> Result<ExpressionAst<K>, ConstructionError> {
        match self {
            Self::Add => permitted(operands, ExpressionAst::<K>::Add),
            Self::Subtract => permitted(operands, ExpressionAst::<K>::Subtract),
            Self::Interact => permitted(operands, ExpressionAst::<K>::Interact),
            Self::InteractFully => permitted(operands, ExpressionAst::<K>::InteractFully),
            Self::Nest => permitted(operands, ExpressionAst::<K>::Nesting),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum StepwiseOperator {
    Sw,
    Sw0,
    Csw,
    Csw0,
}

/// A nonempty sequence of stepwise arguments.
pub(super) struct Stepwise<E> {
    pub(super) operator: StepwiseOperator,
    pub(super) first: E,
    pub(super) rest: Vec<E>,
}

/// A fixed effect with optional varying-slope loadings.
pub(super) struct FixedEffectGroup<K: ExpressionKind> {
    pub(super) group: ExpressionAst<K>,
    pub(super) intercept: bool,
    pub(super) loadings: Vec<Arc<Factor>>,
}

impl<K: ExpressionKind> From<ExpressionAst<K>> for FixedEffectGroup<K> {
    fn from(group: ExpressionAst<K>) -> Self {
        Self {
            group,
            intercept: true,
            loadings: Vec::new(),
        }
    }
}

/// An expression's operand type and permitted operations.
pub(super) trait ExpressionKind: Sized {
    type Operand;

    type Addition: Permission;
    type Subtraction: Permission;
    type Negation: Permission;
    type Interaction: Permission;
    type FullInteraction: Permission;
    type Nesting: Permission;
    type Stepwise: Permission;
    type Child: Permission;
}

/// An expression enum whose [`ExpressionKind`] determines which variants can exist
pub(super) enum ExpressionAst<K: ExpressionKind> {
    Operand(K::Operand),

    Add(K::Addition, Binary<Self>),
    Subtract(K::Subtraction, Binary<Self>),
    Negate(K::Negation, Box<Self>),
    Interact(K::Interaction, Binary<Self>),
    InteractFully(K::FullInteraction, Binary<Self>),
    Nesting(K::Nesting, Binary<Self>),

    Parenthesized(Box<Self>),
    Stepwise(K::Stepwise, Box<Stepwise<Self>>),

    Child(K::Child, Box<FormulaAst>),
}

/// A forbidden expression
impl ExpressionKind for Forbidden {
    type Operand = Forbidden;

    type Addition = Forbidden;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Stepwise = Forbidden;
    type Child = Forbidden;
}

/// An expression without stepwise calls or multistage formulas
pub(super) enum SingleFormulaExpression {}

impl ExpressionKind for SingleFormulaExpression {
    type Operand = crate::Term;

    type Addition = Permitted;
    type Subtraction = Permitted;
    type Negation = Permitted;
    type Interaction = Permitted;
    type FullInteraction = Permitted;
    type Nesting = Permitted;
    type Stepwise = Forbidden;
    type Child = Forbidden;
}

/// An expression over formula terms including stepwise but without multistage terms
pub(super) enum SimpleFormulaExpression {}

impl ExpressionKind for SimpleFormulaExpression {
    type Operand = crate::Term;

    type Addition = Permitted;
    type Subtraction = Permitted;
    type Negation = Permitted;
    type Interaction = Permitted;
    type FullInteraction = Permitted;
    type Nesting = Permitted;
    type Stepwise = Permitted;
    type Child = Forbidden;
}

/// An expression over formula terms including stepwise and multistage
pub(super) enum MultistageFormulaExpression {}

impl ExpressionKind for MultistageFormulaExpression {
    type Operand = ExpressionAst<SimpleFormulaExpression>;

    type Addition = Permitted;
    type Subtraction = Permitted;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Stepwise = Permitted;
    type Child = Permitted;
}

/// An expression over potentially grouped column terms, including stepwise expansion
pub(super) enum GroupingExpression {}

impl ExpressionKind for GroupingExpression {
    type Operand = crate::formula::VariableDefinition;

    type Addition = Permitted;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Permitted;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Stepwise = Permitted;
    type Child = Forbidden;
}

/// An expression over fixed-effect terms, potentially loaded with varying slopes, including stepwise expansion
pub(super) enum FixedEffectExpression {}

impl ExpressionKind for FixedEffectExpression {
    type Operand = FixedEffectGroup<GroupingExpression>;

    type Addition = Permitted;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Stepwise = Permitted;
    type Child = Forbidden;
}

/// An expression over potentially grouped column terms, excluding stepwise expansion
pub(super) enum SingleGroupingExpression {}

impl ExpressionKind for SingleGroupingExpression {
    type Operand = crate::formula::VariableDefinition;

    type Addition = Permitted;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Permitted;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Stepwise = Forbidden;
    type Child = Forbidden;
}

/// An expression over fixed-effect terms, potentially loaded with varying slopes, excluding stepwise expansion
pub(super) enum SingleFixedEffectExpression {}

impl ExpressionKind for SingleFixedEffectExpression {
    type Operand = FixedEffectGroup<SingleGroupingExpression>;

    type Addition = Permitted;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
    type Stepwise = Forbidden;
    type Child = Forbidden;
}

/// Defines an [`ExpressionKind`] with its embedded counterpart
pub(super) trait ParseGrammar: ExpressionKind<Operand: TryFrom<crate::Term>> {
    type Embedded: ExpressionKind<Operand: TryFrom<ExpressionAst<Self>>>;

    fn grouped(
        _group: FixedEffectGroup<Self>,
    ) -> Result<ExpressionAst<Self::Embedded>, ConstructionError> {
        Err(ConstructionError::Unsupported)
    }
}

impl ParseGrammar for SingleFormulaExpression {
    type Embedded = Forbidden;
}

impl ParseGrammar for SimpleFormulaExpression {
    type Embedded = MultistageFormulaExpression;
}

impl ParseGrammar for GroupingExpression {
    type Embedded = FixedEffectExpression;

    fn grouped(
        group: FixedEffectGroup<Self>,
    ) -> Result<ExpressionAst<Self::Embedded>, ConstructionError> {
        Ok(ExpressionAst::Operand(group))
    }
}

impl ParseGrammar for SingleGroupingExpression {
    type Embedded = SingleFixedEffectExpression;

    fn grouped(
        group: FixedEffectGroup<Self>,
    ) -> Result<ExpressionAst<Self::Embedded>, ConstructionError> {
        Ok(ExpressionAst::Operand(group))
    }
}

/// Homogeneous result returned by the Pratt parser.
pub(super) enum ParsedNode<K: ParseGrammar> {
    Flat(ExpressionAst<K>),
    Embedded(ExpressionAst<K::Embedded>),
}

pub(super) type FormulaNode = ParsedNode<SimpleFormulaExpression>;

/// A formula before expansion into estimation alternatives.
pub(super) struct FormulaAst {
    pub(super) left: FormulaNode,
    pub(super) right: FormulaNode,
    pub(super) fixed_effects: Option<ExpressionAst<FixedEffectExpression>>,
}

impl<K: ParseGrammar> ParsedNode<K> {
    fn into_flat(self) -> Result<ExpressionAst<K>, ConstructionError> {
        match self {
            Self::Flat(expression) => Ok(expression),
            Self::Embedded(_) => Err(ConstructionError::Unsupported),
        }
    }

    pub(super) fn parenthesized(self) -> Self {
        match self {
            Self::Flat(expression) => {
                Self::Flat(ExpressionAst::Parenthesized(Box::new(expression)))
            }
            Self::Embedded(expression) => {
                Self::Embedded(ExpressionAst::Parenthesized(Box::new(expression)))
            }
        }
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        match self {
            Self::Flat(expression) => {
                permitted(Box::new(expression), ExpressionAst::<K>::Negate).map(Self::Flat)
            }
            Self::Embedded(expression) => {
                permitted(Box::new(expression), ExpressionAst::<K::Embedded>::Negate)
                    .map(Self::Embedded)
            }
        }
    }

    pub(super) fn child(formula: FormulaAst) -> Result<Self, ConstructionError> {
        permitted(Box::new(formula), ExpressionAst::<K::Embedded>::Child).map(Self::Embedded)
    }

    pub(super) fn from_term(term: crate::Term) -> Result<Self, ConstructionError> {
        let operand = K::Operand::try_from(term).map_err(|_| ConstructionError::Unsupported)?;
        Ok(Self::Flat(ExpressionAst::Operand(operand)))
    }

    pub(super) fn into_embedded(self) -> Result<ExpressionAst<K::Embedded>, ConstructionError> {
        match self {
            Self::Flat(expression) => {
                let operand = <K::Embedded as ExpressionKind>::Operand::try_from(expression)
                    .map_err(|_| ConstructionError::Unsupported)?;

                Ok(ExpressionAst::Operand(operand))
            }
            Self::Embedded(expression) => Ok(expression),
        }
    }

    pub(super) fn binary(
        operator: BinaryOperator,
        left: Self,
        right: Self,
    ) -> Result<Self, ConstructionError> {
        match (left, right) {
            (Self::Flat(left), Self::Flat(right)) => {
                operator.construct(Binary::new(left, right)).map(Self::Flat)
            }
            (left, right) => {
                let operands = Binary::new(left.into_embedded()?, right.into_embedded()?);
                operator.construct(operands).map(Self::Embedded)
            }
        }
    }

    pub(super) fn grouped(
        self,
        intercept: bool,
        first: Arc<Factor>,
        rest: Vec<Arc<Factor>>,
    ) -> Result<Self, ConstructionError> {
        let group = FixedEffectGroup {
            group: self.into_flat()?,
            intercept,
            loadings: std::iter::once(first).chain(rest).collect(),
        };

        K::grouped(group).map(Self::Embedded)
    }

    pub(super) fn stepwise(call: Stepwise<Self>) -> Result<Self, ConstructionError> {
        let has_embedded = std::iter::once(&call.first)
            .chain(call.rest.iter())
            .any(|argument| matches!(argument, Self::Embedded(_)));

        let Stepwise {
            operator,
            first,
            rest,
        } = call;

        if has_embedded {
            let call = Stepwise {
                operator,
                first: first.into_embedded()?,
                rest: rest
                    .into_iter()
                    .map(Self::into_embedded)
                    .collect::<Result<Vec<_>, _>>()?,
            };
            permitted(Box::new(call), ExpressionAst::<K::Embedded>::Stepwise).map(Self::Embedded)
        } else {
            let call = Stepwise {
                operator,
                first: first.into_flat()?,
                rest: rest
                    .into_iter()
                    .map(Self::into_flat)
                    .collect::<Result<Vec<_>, _>>()?,
            };

            permitted(Box::new(call), ExpressionAst::<K>::Stepwise).map(Self::Flat)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Node = ParsedNode<GroupingExpression>;

    fn column(name: &str) -> Node {
        Node::from_term(crate::Term::from(Factor::from(name))).unwrap()
    }

    fn loaded() -> Node {
        column("year")
            .grouped(false, Arc::new(Factor::from("time")), Vec::new())
            .unwrap()
    }

    #[test]
    fn non_nesting_kind_rejects_embedding() {
        let node =
            ParsedNode::<SingleFormulaExpression>::from_term(crate::Term::intercept()).unwrap();
        assert!(matches!(
            node.into_embedded(),
            Err(ConstructionError::Unsupported)
        ));
    }

    #[test]
    fn mixed_binary_wraps_the_flat_subtree_with_default_group_settings() {
        let flat =
            Node::binary(BinaryOperator::Interact, column("firm"), column("sector")).unwrap();
        let Node::Embedded(ExpressionAst::Add(_, operands)) =
            Node::binary(BinaryOperator::Add, flat, loaded()).unwrap()
        else {
            panic!("expected embedded addition");
        };
        let (ExpressionAst::Operand(left), ExpressionAst::Operand(right)) =
            (*operands.left, *operands.right)
        else {
            panic!("expected complete group operands");
        };
        assert!(matches!(left.group, ExpressionAst::Interact(_, _)));
        assert!(left.intercept);
        assert!(left.loadings.is_empty());
        assert!(!right.intercept);
        assert_eq!(right.loadings, vec![Arc::new(Factor::from("time"))]);
    }

    #[test]
    fn mixed_stepwise_wraps_flat_arguments_and_preserves_loaded_arguments() {
        let Node::Embedded(ExpressionAst::Stepwise(_, call)) = Node::stepwise(Stepwise {
            operator: StepwiseOperator::Sw,
            first: column("firm"),
            rest: vec![loaded()],
        })
        .unwrap() else {
            panic!("expected embedded stepwise call");
        };
        let ExpressionAst::Operand(first) = &call.first else {
            panic!("expected a complete first group");
        };
        let [ExpressionAst::Operand(second)] = call.rest.as_slice() else {
            panic!("expected one loaded alternative");
        };
        assert!(first.intercept);
        assert!(first.loadings.is_empty());
        assert!(!second.intercept);
        assert_eq!(second.loadings, vec![Arc::new(Factor::from("time"))]);
    }
}
