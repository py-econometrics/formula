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
    Mvsw,
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

/// Independent operand, algebraic-operator, and expansion choices for an expression.
pub(super) trait ExpressionKind: Sized {
    type Operand: TryFrom<ExpressionAst<Self::Inner>>;
    type Inner: ExpressionKind;
    type Operators: Operations;
    type Stepwise: Permission;
}

/// An expression enum whose [`ExpressionKind`] determines which variants can exist
pub(super) enum ExpressionAst<K: ExpressionKind> {
    Operand(K::Operand),

    Add(<K::Operators as Operations>::Addition, Binary<Self>),
    Subtract(<K::Operators as Operations>::Subtraction, Binary<Self>),
    Negate(<K::Operators as Operations>::Negation, Box<Self>),
    Interact(<K::Operators as Operations>::Interaction, Binary<Self>),
    InteractFully(<K::Operators as Operations>::FullInteraction, Binary<Self>),
    Nesting(<K::Operators as Operations>::Nesting, Binary<Self>),

    Parenthesized(Box<Self>),
    Stepwise(K::Stepwise, Box<Stepwise<Self>>),
}

/// A forbidden expression
impl ExpressionKind for Forbidden {
    type Operand = Forbidden;
    type Inner = Forbidden;
    type Operators = Forbidden;
    type Stepwise = Forbidden;
}

impl Operations for Forbidden {
    type Addition = Forbidden;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
}

pub(super) trait Operations {
    type Addition: Permission;
    type Subtraction: Permission;
    type Negation: Permission;
    type Interaction: Permission;
    type FullInteraction: Permission;
    type Nesting: Permission;
}

pub(super) enum FormulaOperations {}

impl Operations for FormulaOperations {
    type Addition = Permitted;
    type Subtraction = Permitted;
    type Negation = Permitted;
    type Interaction = Permitted;
    type FullInteraction = Permitted;
    type Nesting = Permitted;
}

pub(super) enum MultistageOperations {}

impl Operations for MultistageOperations {
    type Addition = Permitted;
    type Subtraction = Permitted;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
}

pub(super) enum GroupingOperations {}

impl Operations for GroupingOperations {
    type Addition = Permitted;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Permitted;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
}

pub(super) enum FixedEffectOperations {}

impl Operations for FixedEffectOperations {
    type Addition = Permitted;
    type Subtraction = Forbidden;
    type Negation = Forbidden;
    type Interaction = Forbidden;
    type FullInteraction = Forbidden;
    type Nesting = Forbidden;
}

/// An expression layer with independent operands, operators, and stepwise support.
/// `Inner = Forbidden` denotes a layer without an embedded expression domain.
pub(super) struct Layer<O, I, P, S>(std::marker::PhantomData<(O, I, P, S)>);
impl<O, I: ExpressionKind, P: Operations, S: Permission> ExpressionKind for Layer<O, I, P, S>
where
    O: TryFrom<ExpressionAst<I>>,
{
    type Operand = O;
    type Inner = I;
    type Operators = P;
    type Stepwise = S;
}

// Embedding preserves the complete inner tree; operand expansion decides how
// it participates in the outer domain.
impl<I, O, P, S> From<ExpressionAst<I>> for ExpressionAst<Layer<O, I, P, S>>
where
    I: ExpressionKind,
    O: From<ExpressionAst<I>>,
    P: Operations,
    S: Permission,
{
    fn from(inner: ExpressionAst<I>) -> Self {
        Self::Operand(O::from(inner))
    }
}

/// A multistage operand is either an embedded simple expression or a child formula.
pub(super) enum MultistageOperand<I: ExpressionKind> {
    Expression(Box<ExpressionAst<I>>),
    Child(Box<FormulaAst>),
}
impl<I: ExpressionKind> From<ExpressionAst<I>> for MultistageOperand<I> {
    fn from(value: ExpressionAst<I>) -> Self {
        Self::Expression(Box::new(value))
    }
}
impl<I: ExpressionKind> From<FormulaAst> for MultistageOperand<I> {
    fn from(value: FormulaAst) -> Self {
        Self::Child(Box::new(value))
    }
}

pub(super) type SingleFormulaExpression =
    Layer<crate::Term, Forbidden, FormulaOperations, Forbidden>;
pub(super) type SimpleFormulaExpression =
    Layer<crate::Term, Forbidden, FormulaOperations, Permitted>;
pub(super) type GroupingExpression =
    Layer<crate::formula::VariableDefinition, Forbidden, GroupingOperations, Permitted>;
pub(super) type SingleGroupingExpression =
    Layer<crate::formula::VariableDefinition, Forbidden, GroupingOperations, Forbidden>;
pub(super) type MultistageFormulaExpression = Layer<
    MultistageOperand<SimpleFormulaExpression>,
    SimpleFormulaExpression,
    MultistageOperations,
    Permitted,
>;
pub(super) type FixedEffectExpression = Layer<
    FixedEffectGroup<GroupingExpression>,
    GroupingExpression,
    FixedEffectOperations,
    Permitted,
>;

pub(super) type SingleFixedEffectExpression = Layer<
    FixedEffectGroup<SingleGroupingExpression>,
    SingleGroupingExpression,
    FixedEffectOperations,
    Forbidden,
>;

// Leaf operands cannot embed expressions. This is the same embedding interface
// as nested operands, with an uninhabited source layer.
impl TryFrom<ExpressionAst<Forbidden>> for crate::Term {
    type Error = ConstructionError;
    fn try_from(_: ExpressionAst<Forbidden>) -> Result<Self, Self::Error> {
        Err(ConstructionError::Unsupported)
    }
}
impl TryFrom<ExpressionAst<Forbidden>> for crate::formula::VariableDefinition {
    type Error = ConstructionError;
    fn try_from(_: ExpressionAst<Forbidden>) -> Result<Self, Self::Error> {
        Err(ConstructionError::Unsupported)
    }
}

impl<K: ExpressionKind> ExpressionAst<K> {
    pub(super) fn try_operand<T>(input: T) -> Result<Self, ConstructionError>
    where
        K::Operand: TryFrom<T>,
    {
        K::Operand::try_from(input)
            .map(Self::Operand)
            .map_err(|_| ConstructionError::Unsupported)
    }
}

/// Homogeneous result returned by the Pratt parser.
pub(super) enum ParsedNode<K: ExpressionKind> {
    Inner(ExpressionAst<K::Inner>),
    Outer(ExpressionAst<K>),
}

pub(super) type FormulaNode = ParsedNode<MultistageFormulaExpression>;

/// A formula before expansion into estimation alternatives.
pub(super) struct FormulaAst {
    pub(super) left: FormulaNode,
    pub(super) right: FormulaNode,
    pub(super) fixed_effects: Option<ExpressionAst<FixedEffectExpression>>,
}

impl<K: ExpressionKind> ParsedNode<K> {
    pub(super) fn into_inner(self) -> Result<ExpressionAst<K::Inner>, ConstructionError> {
        match self {
            Self::Inner(expression) => Ok(expression),
            Self::Outer(_) => Err(ConstructionError::Unsupported),
        }
    }

    pub(super) fn parenthesized(self) -> Self {
        match self {
            Self::Inner(expression) => {
                Self::Inner(ExpressionAst::Parenthesized(Box::new(expression)))
            }
            Self::Outer(expression) => {
                Self::Outer(ExpressionAst::Parenthesized(Box::new(expression)))
            }
        }
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        match self {
            Self::Inner(expression) => {
                permitted(Box::new(expression), ExpressionAst::<K::Inner>::Negate).map(Self::Inner)
            }
            Self::Outer(expression) => {
                permitted(Box::new(expression), ExpressionAst::<K>::Negate).map(Self::Outer)
            }
        }
    }

    pub(super) fn into_outer(self) -> Result<ExpressionAst<K>, ConstructionError> {
        match self {
            Self::Inner(expression) => ExpressionAst::try_operand(expression),
            Self::Outer(expression) => Ok(expression),
        }
    }

    pub(super) fn binary(
        operator: BinaryOperator,
        left: Self,
        right: Self,
    ) -> Result<Self, ConstructionError> {
        match (left, right) {
            (Self::Inner(left), Self::Inner(right)) => operator
                .construct(Binary::new(left, right))
                .map(Self::Inner),
            (left, right) => {
                let operands = Binary::new(left.into_outer()?, right.into_outer()?);
                operator.construct(operands).map(Self::Outer)
            }
        }
    }

    pub(super) fn stepwise(call: Stepwise<Self>) -> Result<Self, ConstructionError> {
        let has_outer = std::iter::once(&call.first)
            .chain(call.rest.iter())
            .any(|argument| matches!(argument, Self::Outer(_)));

        let Stepwise {
            operator,
            first,
            rest,
        } = call;

        if has_outer {
            let call = Stepwise {
                operator,
                first: first.into_outer()?,
                rest: rest
                    .into_iter()
                    .map(Self::into_outer)
                    .collect::<Result<Vec<_>, _>>()?,
            };
            permitted(Box::new(call), ExpressionAst::<K>::Stepwise).map(Self::Outer)
        } else {
            let call = Stepwise {
                operator,
                first: first.into_inner()?,
                rest: rest
                    .into_iter()
                    .map(Self::into_inner)
                    .collect::<Result<Vec<_>, _>>()?,
            };

            permitted(Box::new(call), ExpressionAst::<K::Inner>::Stepwise).map(Self::Inner)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Node = ParsedNode<FixedEffectExpression>;

    fn column(name: &str) -> Node {
        Node::Inner(ExpressionAst::try_operand(crate::Term::from(Factor::from(name))).unwrap())
    }

    fn loaded() -> Node {
        Node::Outer(ExpressionAst::Operand(FixedEffectGroup {
            group: column("year").into_inner().unwrap(),
            intercept: false,
            loadings: vec![Arc::new(Factor::from("time"))],
        }))
    }

    #[test]
    fn leaf_node_has_no_inner_expression() {
        let node = ExpressionAst::<SingleFormulaExpression>::try_operand(crate::Term::intercept())
            .unwrap();
        assert!(matches!(
            ParsedNode::<SingleFormulaExpression>::Outer(node).into_inner(),
            Err(ConstructionError::Unsupported)
        ));
    }

    #[test]
    fn mixed_binary_wraps_the_flat_subtree_with_default_group_settings() {
        let flat =
            Node::binary(BinaryOperator::Interact, column("firm"), column("sector")).unwrap();
        let Node::Outer(ExpressionAst::Add(_, operands)) =
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
        let Node::Outer(ExpressionAst::Stepwise(_, call)) = Node::stepwise(Stepwise {
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

#[cfg(test)]
mod layers {
    use super::*;

    // A third layer with a new operand type: no Child or Grouped vocabulary.
    struct Envelope(ExpressionAst<FixedEffectExpression>);
    impl From<ExpressionAst<FixedEffectExpression>> for Envelope {
        fn from(value: ExpressionAst<FixedEffectExpression>) -> Self {
            Self(value)
        }
    }
    type Enveloped = Layer<Envelope, FixedEffectExpression, FixedEffectOperations, Forbidden>;

    #[test]
    fn new_layer_embeds_an_existing_layer_without_changing_the_ast() {
        let grouping = ExpressionAst::<GroupingExpression>::try_operand(crate::Term::from(
            Factor::from("firm"),
        ))
        .unwrap_or_else(|_| panic!("column"));
        let fixed: ExpressionAst<FixedEffectExpression> = grouping.into();
        let envelope: ExpressionAst<Enveloped> = fixed.into();
        let ExpressionAst::Operand(Envelope(ExpressionAst::Operand(group))) = envelope else {
            panic!("wrong representation");
        };
        assert!(group.intercept);
        assert!(group.loadings.is_empty());
    }
}
