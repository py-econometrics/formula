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

impl<E: Layer> TryFrom<ExpressionAst<E>> for Forbidden {
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
    fn construct<K: Layer>(
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
pub(super) struct FixedEffect<K: Layer> {
    pub(super) group: ExpressionAst<K>,
}
impl<K: Layer> From<ExpressionAst<K>> for FixedEffect<K> {
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
pub(super) trait Embedding<I: Layer>: TryFrom<ExpressionAst<I>> {
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

/// A layer of a formula expression: support for operands, operators and stepwise syntax.
pub(super) trait Layer: Sized {
    type Operand: Embedding<Self::Inner>;
    type Nested;
    type Inner: Layer;
    type Operators: Operations;
    type Stepwise: Permission;
}

/// An expression in layer, which determines the variants that can exist.
pub(super) enum ExpressionAst<K: Layer> {
    Operand(K::Operand),
    Nested(Box<K::Nested>),

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
impl Layer for Forbidden {
    type Operand = Forbidden;
    type Nested = Forbidden;
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

/// A [`Layer`] given by its operand, inner-layer, operator,
/// stepwise and nested value. `Inner = Forbidden` means nothing is embedded.
pub(super) struct LayerSpec<O, I, P, S, N = Forbidden>(std::marker::PhantomData<(O, I, P, S, N)>);
impl<O, I: Layer, P: Operations, S: Permission, N> Layer for LayerSpec<O, I, P, S, N>
where
    O: Embedding<I>,
{
    type Operand = O;
    type Nested = N;
    type Inner = I;
    type Operators = P;
    type Stepwise = S;
}

// Embedding preserves the complete inner tree; operand expansion decides how
// it participates in the outer domain.
impl<I, O, P, S, N> From<ExpressionAst<I>> for ExpressionAst<LayerSpec<O, I, P, S, N>>
where
    I: Layer,
    O: Embedding<I> + From<ExpressionAst<I>>,
    P: Operations,
    S: Permission,
{
    fn from(inner: ExpressionAst<I>) -> Self {
        Self::Operand(O::from(inner))
    }
}

impl<I: Layer> Embedding<I> for ExpressionAst<I> {
    fn is_inner(&self) -> bool {
        true
    }
    fn into_inner(self) -> Result<ExpressionAst<I>, Self> {
        Ok(self)
    }
}
impl<I: Layer> Embedding<I> for FixedEffect<I> {
    fn is_inner(&self) -> bool {
        true
    }
    fn into_inner(self) -> Result<ExpressionAst<I>, Self> {
        Ok(self.group)
    }
}

pub(super) type SingleFormula = LayerSpec<crate::Term, Forbidden, FormulaOperations, Forbidden>;
pub(super) type SimpleFormula = LayerSpec<crate::Term, Forbidden, FormulaOperations, Permitted>;
pub(super) type Grouping =
    LayerSpec<crate::formula::VariableDefinition, Forbidden, GroupingOperations, Permitted>;
pub(super) type SingleGrouping =
    LayerSpec<crate::formula::VariableDefinition, Forbidden, GroupingOperations, Forbidden>;
pub(super) type MultistageFormula = LayerSpec<
    ExpressionAst<SimpleFormula>,
    SimpleFormula,
    MultistageOperations,
    Permitted,
    FormulaAst,
>;
pub(super) type FixedEffects = LayerSpec<
    FixedEffect<Grouping>,
    Grouping,
    FixedEffectOperations,
    Permitted,
    Loaded<FixedEffect<Grouping>>,
>;

pub(super) type SingleFixedEffects = LayerSpec<
    FixedEffect<SingleGrouping>,
    SingleGrouping,
    FixedEffectOperations,
    Forbidden,
    Loaded<FixedEffect<SingleGrouping>>,
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

impl<K: Layer> ExpressionAst<K> {
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
    pub(super) left: ExpressionAst<MultistageFormula>,
    pub(super) right: ExpressionAst<MultistageFormula>,
    pub(super) fixed_effects: Option<ExpressionAst<FixedEffects>>,
}

impl<K: Layer> ExpressionAst<K> {
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
            Ok(inner) => inner.parenthesized().and_then(Self::try_operand),
            Err(outer) => Ok(Self::Parenthesized(Box::new(outer))),
        }
    }

    pub(super) fn negate(self) -> Result<Self, ConstructionError> {
        match self.try_into_inner() {
            Ok(inner) => inner.negate().and_then(Self::try_operand),
            Err(outer) => permitted(Box::new(outer), Self::Negate),
        }
    }

    pub(super) fn binary(
        operator: BinaryOperator,
        left: Self,
        right: Self,
    ) -> Result<Self, ConstructionError> {
        if left.is_inner() && right.is_inner() {
            ExpressionAst::<K::Inner>::binary(operator, left.into_inner()?, right.into_inner()?)
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
            ExpressionAst::<K::Inner>::stepwise(inner_call).and_then(Self::try_operand)
        } else {
            permitted(Box::new(call), Self::Stepwise)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Node = ExpressionAst<FixedEffects>;

    fn column(name: &str) -> Node {
        Node::try_operand(
            ExpressionAst::<Grouping>::try_operand(crate::Term::from(Factor::from(name))).unwrap(),
        )
        .unwrap()
    }

    fn loaded() -> Node {
        ExpressionAst::Nested(Box::new(Loaded {
            group: column("year").into_inner().unwrap().into(),
            intercept: false,
            first_loading: Arc::new(Factor::from("time")),
            rest_loadings: Vec::new(),
        }))
    }

    #[test]
    fn leaf_node_has_no_inner_expression() {
        let node = ExpressionAst::<SingleFormula>::try_operand(crate::Term::intercept()).unwrap();
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
        let (ExpressionAst::Operand(left), ExpressionAst::Nested(right)) =
            (*operands.left, *operands.right)
        else {
            panic!("expected complete group operands");
        };
        assert!(matches!(left.group, ExpressionAst::Interact(_, _)));
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
        let [ExpressionAst::Nested(second)] = call.rest.as_slice() else {
            panic!("expected one loaded alternative");
        };
        assert!(matches!(first.group, ExpressionAst::Operand(_)));
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
    struct Envelope(ExpressionAst<FixedEffects>);
    impl From<ExpressionAst<FixedEffects>> for Envelope {
        fn from(value: ExpressionAst<FixedEffects>) -> Self {
            Self(value)
        }
    }
    impl Embedding<FixedEffects> for Envelope {
        fn is_inner(&self) -> bool {
            true
        }
        fn into_inner(self) -> Result<ExpressionAst<FixedEffects>, Self> {
            Ok(self.0)
        }
    }
    type Enveloped = LayerSpec<Envelope, FixedEffects, FixedEffectOperations, Forbidden>;

    #[test]
    fn new_layer_embeds_an_existing_layer_without_changing_the_ast() {
        let grouping =
            ExpressionAst::<Grouping>::try_operand(crate::Term::from(Factor::from("firm")))
                .unwrap_or_else(|_| panic!("column"));
        let fixed: ExpressionAst<FixedEffects> = grouping.into();
        let envelope: ExpressionAst<Enveloped> = fixed.into();
        let ExpressionAst::Operand(Envelope(ExpressionAst::Operand(group))) = envelope else {
            panic!("wrong representation");
        };
        assert!(matches!(group.group, ExpressionAst::Operand(_)));
    }
}
