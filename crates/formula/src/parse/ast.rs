use crate::formula::VaryingSlopes;

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
    fn construct<L: Layer>(
        self,
        operands: Binary<ExpressionAst<L>>,
    ) -> Result<ExpressionAst<L>, ConstructionError> {
        match self {
            Self::Add => permitted(operands, ExpressionAst::<L>::Add),
            Self::Subtract => permitted(operands, ExpressionAst::<L>::Subtract),
            Self::Interact => permitted(operands, ExpressionAst::<L>::Interact),
            Self::InteractFully => permitted(operands, ExpressionAst::<L>::InteractFully),
            Self::Nest => permitted(operands, ExpressionAst::<L>::Nesting),
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

/// A layer of a formula expression.
///
/// A layer is defined by its atomic operand and operators. It may also embed an inner layer,
/// and optionally annotate the inner layer.
pub(super) trait Layer {
    type Atom;
    type Operators: Operations;
    type Inner: Compose;
    type Annotation;
    type Stepwise: Permission;
}

/// An expression in layer `L` determining which variants can exist.
pub(super) enum ExpressionAst<L: Layer> {
    Atom(L::Atom),

    Embedded(L::Inner, Option<L::Annotation>),

    Add(<L::Operators as Operations>::Addition, Binary<Self>),
    Subtract(<L::Operators as Operations>::Subtraction, Binary<Self>),
    Negate(<L::Operators as Operations>::Negation, Box<Self>),
    Interact(<L::Operators as Operations>::Interaction, Binary<Self>),
    InteractFully(<L::Operators as Operations>::FullInteraction, Binary<Self>),
    Nesting(<L::Operators as Operations>::Nesting, Binary<Self>),

    Parenthesized(Box<Self>),
    Stepwise(L::Stepwise, Box<Stepwise<Self>>),
}

/// A description of how [`Layer`]s can be composed.
pub(super) trait Compose: Sized {
    fn parenthesized(self) -> Self;
    fn negate(self) -> Result<Self, ConstructionError>;
    fn binary(operator: BinaryOperator, left: Self, right: Self)
    -> Result<Self, ConstructionError>;
    fn stepwise(call: Stepwise<Self>) -> Result<Self, ConstructionError>;
}

impl Compose for Forbidden {
    fn parenthesized(self) -> Self {
        match self {}
    }
    fn negate(self) -> Result<Self, ConstructionError> {
        match self {}
    }
    fn binary(_: BinaryOperator, left: Self, _: Self) -> Result<Self, ConstructionError> {
        match left {}
    }
    fn stepwise(call: Stepwise<Self>) -> Result<Self, ConstructionError> {
        match call.first {}
    }
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

/// A formula with stepwise calls
pub(super) enum SimpleFormula {}
impl Layer for SimpleFormula {
    type Atom = crate::Term;
    type Operators = FormulaOperations;
    type Inner = Forbidden;
    type Annotation = Forbidden;
    type Stepwise = Permitted;
}

/// A formula including multistage child formulas
pub(super) enum MultistageFormula {}
impl Layer for MultistageFormula {
    type Atom = Box<FormulaAst>;
    type Operators = MultistageOperations;
    type Inner = ExpressionAst<SimpleFormula>;
    type Annotation = Forbidden;
    type Stepwise = Permitted;
}

/// A simple formula without stepwise calls
pub(super) enum SingleSimpleFormula {}
impl Layer for SingleSimpleFormula {
    type Atom = crate::Term;
    type Operators = FormulaOperations;
    type Inner = Forbidden;
    type Annotation = Forbidden;
    type Stepwise = Forbidden;
}

/// Grouping of bare variables
pub(super) enum VariableGrouping {}
impl Layer for VariableGrouping {
    type Atom = crate::formula::VariableDefinition;
    type Operators = GroupingOperations;
    type Inner = Forbidden;
    type Annotation = Forbidden;
    type Stepwise = Permitted;
}

/// Grouping of bare variables excluding stepwise calls
pub(super) enum SingleVariableGrouping {}
impl Layer for SingleVariableGrouping {
    type Atom = crate::formula::VariableDefinition;
    type Operators = GroupingOperations;
    type Inner = Forbidden;
    type Annotation = Forbidden;
    type Stepwise = Forbidden;
}

/// Grouping of bare variables with optional varying slopes
pub(super) enum FixedEffects {}
impl Layer for FixedEffects {
    type Atom = Forbidden;
    type Operators = FixedEffectOperations;
    type Inner = ExpressionAst<VariableGrouping>;
    type Annotation = VaryingSlopes;
    type Stepwise = Permitted;
}

/// Grouping of bare variables with optional varying slopes but without stepwise calls
pub(super) enum SingleFixedEffects {}
impl Layer for SingleFixedEffects {
    type Atom = Forbidden;
    type Operators = FixedEffectOperations;
    type Inner = ExpressionAst<SingleVariableGrouping>;
    type Annotation = VaryingSlopes;
    type Stepwise = Forbidden;
}

/// A formula before expansion into estimation alternatives.
pub(super) struct FormulaAst {
    pub(super) left: ExpressionAst<MultistageFormula>,
    pub(super) right: ExpressionAst<MultistageFormula>,
    pub(super) fixed_effects: Option<ExpressionAst<FixedEffects>>,
}

/// Implements the lifecycle of an [`ExpressionAst`]:
/// atom -> operators -> embed -> annotate -> next layer ->
impl<L: Layer> ExpressionAst<L> {
    pub(super) fn embed(inner: L::Inner) -> Self {
        Self::Embedded(inner, None)
    }

    pub(super) fn into_inner(self) -> Result<L::Inner, Self> {
        match self {
            Self::Embedded(inner, None) => Ok(inner),
            outer => Err(outer),
        }
    }

    pub(super) fn annotate(self, annotation: L::Annotation) -> Result<Self, ConstructionError> {
        match self {
            Self::Embedded(inner, None) => Ok(Self::Embedded(inner, Some(annotation))),
            _ => Err(ConstructionError::Unsupported),
        }
    }
}

/// [`Compose`] operators of an [`ExpressionAst`]
impl<L: Layer> Compose for ExpressionAst<L> {
    fn parenthesized(self) -> Self {
        match self.into_inner() {
            Ok(inner) => Self::embed(inner.parenthesized()),
            Err(outer) => Self::Parenthesized(Box::new(outer)),
        }
    }

    fn negate(self) -> Result<Self, ConstructionError> {
        match self.into_inner() {
            Ok(inner) => inner.negate().map(Self::embed),
            Err(outer) => permitted(Box::new(outer), Self::Negate),
        }
    }

    fn binary(
        operator: BinaryOperator,
        left: Self,
        right: Self,
    ) -> Result<Self, ConstructionError> {
        match (left, right) {
            (Self::Embedded(left, None), Self::Embedded(right, None)) => {
                L::Inner::binary(operator, left, right).map(Self::embed)
            }
            (left, right) => operator.construct(Binary::new(left, right)),
        }
    }

    fn stepwise(call: Stepwise<Self>) -> Result<Self, ConstructionError> {
        let is_inner = |argument: &Self| matches!(argument, Self::Embedded(_, None));
        if !std::iter::once(&call.first).chain(&call.rest).all(is_inner) {
            return permitted(Box::new(call), Self::Stepwise);
        }
        let inner = |argument: Self| {
            argument
                .into_inner()
                .map_err(|_| ConstructionError::Unsupported)
        };
        let call = Stepwise {
            operator: call.operator,
            first: inner(call.first)?,
            rest: call.rest.into_iter().map(inner).collect::<Result<_, _>>()?,
        };
        L::Inner::stepwise(call).map(Self::embed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Factor;
    use std::sync::Arc;

    type Node = ExpressionAst<FixedEffects>;

    fn column(name: &str) -> Node {
        let definition =
            crate::formula::VariableDefinition::try_from(crate::Term::from(Factor::from(name)))
                .unwrap();
        Node::embed(ExpressionAst::Atom(definition))
    }

    fn slopes(loading: &str) -> VaryingSlopes {
        VaryingSlopes {
            intercept: false,
            first: Arc::new(Factor::from(loading)),
            rest: Vec::new(),
        }
    }

    fn loaded() -> Node {
        column("year").annotate(slopes("time")).unwrap()
    }

    #[test]
    fn atoms_have_no_inner_expression() {
        let node = ExpressionAst::<SingleSimpleFormula>::Atom(crate::Term::intercept());
        assert!(node.into_inner().is_err());
    }

    #[test]
    fn annotations_apply_once_and_only_to_unannotated_embeddings() {
        assert!(loaded().into_inner().is_err());
        assert!(loaded().annotate(slopes("u")).is_err());
        let sum = Node::binary(BinaryOperator::Add, loaded(), column("firm")).unwrap();
        assert!(sum.annotate(slopes("u")).is_err());
    }

    #[test]
    fn binary_operators_combine_in_the_innermost_shared_layer() {
        let interaction =
            Node::binary(BinaryOperator::Interact, column("firm"), column("sector")).unwrap();
        assert!(matches!(
            interaction,
            ExpressionAst::Embedded(ExpressionAst::Interact(_, _), None)
        ));

        let ExpressionAst::Add(_, operands) =
            Node::binary(BinaryOperator::Add, interaction, loaded()).unwrap()
        else {
            panic!("expected an addition in the fixed-effect layer");
        };
        assert!(matches!(
            *operands.left,
            ExpressionAst::Embedded(ExpressionAst::Interact(_, _), None)
        ));
        assert!(matches!(
            *operands.right,
            ExpressionAst::Embedded(_, Some(_))
        ));

        // A loaded group stays in the fixed-effect layer, which forbids interaction.
        assert!(Node::binary(BinaryOperator::Interact, loaded(), column("firm")).is_err());
    }

    #[test]
    fn stepwise_calls_combine_in_the_innermost_shared_layer() {
        let call = |rest: Vec<Node>| {
            Node::stepwise(Stepwise {
                operator: StepwiseOperator::Sw,
                first: column("firm"),
                rest,
            })
            .unwrap()
        };
        assert!(matches!(
            call(vec![column("year")]),
            ExpressionAst::Embedded(ExpressionAst::Stepwise(_, _), None)
        ));

        let ExpressionAst::Stepwise(_, mixed) = call(vec![loaded()]) else {
            panic!("expected a stepwise call in the fixed-effect layer");
        };
        assert!(matches!(mixed.first, ExpressionAst::Embedded(_, None)));
        assert!(matches!(
            mixed.rest.as_slice(),
            [ExpressionAst::Embedded(_, Some(_))]
        ));
    }
}
