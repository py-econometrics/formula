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

/// An unloaded fixed effect, with its grouping algebra still intact.
pub(super) struct FixedEffect<K: ExpressionKind> {
    pub(super) group: ExpressionAst<K>,
}
impl<K: ExpressionKind> From<ExpressionAst<K>> for FixedEffect<K> {
    fn from(group: ExpressionAst<K>) -> Self {
        Self { group }
    }
}

/// Loading is a wrapper around an unloaded fixed effect.
pub(super) struct Loaded<T> {
    pub(super) group: T,
    pub(super) intercept: bool,
    pub(super) first_loading: Arc<Factor>,
    pub(super) rest_loadings: Vec<Arc<Factor>>,
}

impl<T> Loaded<T> {
    pub(super) fn loadings(&self) -> impl Iterator<Item = &Arc<Factor>> {
        std::iter::once(&self.first_loading).chain(self.rest_loadings.iter())
    }
}

pub(super) type FixedEffectGroup<K> = Loaded<FixedEffect<K>>;

/// Embedding is reversible: the parser can recover an inner tree without
/// maintaining a second representation of the expression's current layer.
pub(super) trait Embedding<I: ExpressionKind>: TryFrom<ExpressionAst<I>> {
    fn is_inner(&self) -> bool {
        false
    }
    fn into_inner(self) -> Result<ExpressionAst<I>, Self> {
        Err(self)
    }
}
impl Embedding<Forbidden> for crate::Term {}
impl Embedding<Forbidden> for crate::formula::VariableDefinition {}
impl Embedding<Forbidden> for Forbidden {}

/// Independent operand, algebraic-operator, and expansion choices for an expression.
pub(super) trait ExpressionKind: Sized {
    type Operand: Embedding<Self::Inner>;
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
    O: Embedding<I>,
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
    O: Embedding<I> + From<ExpressionAst<I>>,
    P: Operations,
    S: Permission,
{
    fn from(inner: ExpressionAst<I>) -> Self {
        Self::Operand(O::from(inner))
    }
}

/// One operand lifecycle for every layer: preserve an inner tree, or construct
/// a domain-specific nested value. No syntax or domain names in this enum.
pub(super) enum LayerOperand<B, N> {
    Expression(B),
    Nested(Box<N>),
}
impl<I: ExpressionKind, B: From<ExpressionAst<I>>, N> From<ExpressionAst<I>>
    for LayerOperand<B, N>
{
    fn from(value: ExpressionAst<I>) -> Self {
        Self::Expression(B::from(value))
    }
}

impl<K: ExpressionKind> From<FixedEffect<K>> for ExpressionAst<K> {
    fn from(effect: FixedEffect<K>) -> Self {
        effect.group
    }
}
impl<I: ExpressionKind, B, N> Embedding<I> for LayerOperand<B, N>
where
    B: From<ExpressionAst<I>> + Into<ExpressionAst<I>>,
{
    fn is_inner(&self) -> bool {
        matches!(self, Self::Expression(_))
    }
    fn into_inner(self) -> Result<ExpressionAst<I>, Self> {
        match self {
            Self::Expression(base) => Ok(base.into()),
            other => Err(other),
        }
    }
}

pub(super) type MultistageOperand<I> = LayerOperand<ExpressionAst<I>, FormulaAst>;
pub(super) type FixedEffectOperand<I> = LayerOperand<FixedEffect<I>, FixedEffectGroup<I>>;

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
    FixedEffectOperand<GroupingExpression>,
    GroupingExpression,
    FixedEffectOperations,
    Permitted,
>;

pub(super) type SingleFixedEffectExpression = Layer<
    FixedEffectOperand<SingleGroupingExpression>,
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

/// A formula before expansion into estimation alternatives.
pub(super) struct FormulaAst {
    pub(super) left: ExpressionAst<MultistageFormulaExpression>,
    pub(super) right: ExpressionAst<MultistageFormulaExpression>,
    pub(super) fixed_effects: Option<ExpressionAst<FixedEffectExpression>>,
}

impl<K: ExpressionKind> ExpressionAst<K> {
    fn is_inner(&self) -> bool {
        matches!(self, Self::Operand(operand) if operand.is_inner())
    }

    fn try_into_inner(self) -> Result<ExpressionAst<K::Inner>, Self> {
        match self {
            Self::Operand(operand) => operand.into_inner().map_err(Self::Operand),
            other => Err(other),
        }
    }

    pub(super) fn into_inner(self) -> Result<ExpressionAst<K::Inner>, ConstructionError> {
        self.try_into_inner()
            .map_err(|_| ConstructionError::Unsupported)
    }

    pub(super) fn parenthesized(self) -> Result<Self, ConstructionError> {
        match self.try_into_inner() {
            Ok(inner) => Self::try_operand(ExpressionAst::Parenthesized(Box::new(inner))),
            Err(outer) => Ok(Self::Parenthesized(Box::new(outer))),
        }
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        match self.try_into_inner() {
            Ok(inner) => permitted(Box::new(inner), ExpressionAst::<K::Inner>::Negate)
                .and_then(Self::try_operand),
            Err(outer) => permitted(Box::new(outer), Self::Negate),
        }
    }

    pub(super) fn binary(
        operator: BinaryOperator,
        left: Self,
        right: Self,
    ) -> Result<Self, ConstructionError> {
        if left.is_inner() && right.is_inner() {
            operator
                .construct(Binary::new(left.into_inner()?, right.into_inner()?))
                .and_then(Self::try_operand)
        } else {
            operator.construct(Binary::new(left, right))
        }
    }

    pub(super) fn stepwise(call: Stepwise<Self>) -> Result<Self, ConstructionError> {
        let all_inner = std::iter::once(&call.first)
            .chain(call.rest.iter())
            .all(|argument| argument.is_inner());
        if all_inner {
            let inner_call = Stepwise {
                operator: call.operator,
                first: call.first.into_inner()?,
                rest: call
                    .rest
                    .into_iter()
                    .map(Self::into_inner)
                    .collect::<Result<_, _>>()?,
            };
            permitted(Box::new(inner_call), ExpressionAst::<K::Inner>::Stepwise)
                .and_then(Self::try_operand)
        } else {
            permitted(Box::new(call), Self::Stepwise)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Node = ExpressionAst<FixedEffectExpression>;

    fn column(name: &str) -> Node {
        Node::try_operand(
            ExpressionAst::<GroupingExpression>::try_operand(crate::Term::from(Factor::from(name)))
                .unwrap(),
        )
        .unwrap()
    }

    fn loaded() -> Node {
        ExpressionAst::Operand(LayerOperand::Nested(Box::new(FixedEffectGroup {
            group: column("year").into_inner().unwrap().into(),
            intercept: false,
            first_loading: Arc::new(Factor::from("time")),
            rest_loadings: Vec::new(),
        })))
    }

    #[test]
    fn leaf_node_has_no_inner_expression() {
        let node = ExpressionAst::<SingleFormulaExpression>::try_operand(crate::Term::intercept())
            .unwrap();
        assert!(matches!(
            node.into_inner(),
            Err(ConstructionError::Unsupported)
        ));
    }

    #[test]
    fn mixed_binary_wraps_the_flat_subtree_with_default_group_settings() {
        let flat =
            Node::binary(BinaryOperator::Interact, column("firm"), column("sector")).unwrap();
        let ExpressionAst::Add(_, operands) =
            Node::binary(BinaryOperator::Add, flat, loaded()).unwrap()
        else {
            panic!("expected embedded addition");
        };
        let (ExpressionAst::Operand(left), ExpressionAst::Operand(right)) =
            (*operands.left, *operands.right)
        else {
            panic!("expected complete group operands");
        };
        let LayerOperand::Expression(left) = left else {
            panic!("plain inner");
        };
        assert!(matches!(left.group, ExpressionAst::Interact(_, _)));
        let LayerOperand::Nested(right) = right else {
            panic!("loaded inner");
        };
        assert!(!right.intercept);
        assert_eq!(
            right.loadings().cloned().collect::<Vec<_>>(),
            vec![Arc::new(Factor::from("time"))]
        );
    }

    #[test]
    fn mixed_stepwise_wraps_flat_arguments_and_preserves_loaded_arguments() {
        let ExpressionAst::Stepwise(_, call) = Node::stepwise(Stepwise {
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
        assert!(matches!(first, LayerOperand::Expression(_)));
        let LayerOperand::Nested(second) = second else {
            panic!("loaded alternative");
        };
        assert!(!second.intercept);
        assert_eq!(
            second.loadings().cloned().collect::<Vec<_>>(),
            vec![Arc::new(Factor::from("time"))]
        );
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
    impl Embedding<FixedEffectExpression> for Envelope {
        fn is_inner(&self) -> bool {
            true
        }
        fn into_inner(self) -> Result<ExpressionAst<FixedEffectExpression>, Self> {
            Ok(self.0)
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
        let ExpressionAst::Operand(Envelope(ExpressionAst::Operand(LayerOperand::Expression(
            group,
        )))) = envelope
        else {
            panic!("wrong representation");
        };
        assert!(matches!(group.group, ExpressionAst::Operand(_)));
    }
}
