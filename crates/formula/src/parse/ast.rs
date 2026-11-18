use crate::formula::VaryingSlopes;

pub(super) struct Permitted;
pub(super) enum Forbidden {}

pub(super) trait Permission: Sized {
    fn permit() -> Option<Self>;
}

impl Permission for Permitted {
    fn permit() -> Option<Self> {
        Some(Self)
    }
}

impl Permission for Forbidden {
    fn permit() -> Option<Self> {
        None
    }
}

/// Apply the constructor if it is permitted, or hand the value back
pub(super) fn permitted<P: Permission, T, E>(
    value: T,
    constructor: impl FnOnce(P, T) -> E,
) -> Result<E, T> {
    match P::permit() {
        Some(permission) => Ok(constructor(permission, value)),
        None => Err(value),
    }
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

#[derive(Clone, Copy)]
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
        left: ExpressionAst<L>,
        right: ExpressionAst<L>,
    ) -> Result<ExpressionAst<L>, (ExpressionAst<L>, ExpressionAst<L>)> {
        let operands = Binary::new(left, right);
        match self {
            Self::Add => permitted(operands, ExpressionAst::<L>::Add),
            Self::Subtract => permitted(operands, ExpressionAst::<L>::Subtract),
            Self::Interact => permitted(operands, ExpressionAst::<L>::Interact),
            Self::InteractFully => permitted(operands, ExpressionAst::<L>::InteractFully),
            Self::Nest => permitted(operands, ExpressionAst::<L>::Nesting),
        }
        .map_err(|operands| (*operands.left, *operands.right))
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
    /// `(a + b)^n`: the full interaction of `n` copies, so every interaction of
    /// up to `n` distinct factors. Permitted wherever `*` is.
    Power(
        <L::Operators as Operations>::FullInteraction,
        Box<Self>,
        u32,
    ),

    Parenthesized(Box<Self>),
    /// A stepwise call over a nonempty sequence of arguments.
    Stepwise(L::Stepwise, StepwiseOperator, Box<Self>, Vec<Self>),
}

/// A description of how [`Layer`]s can be composed.
pub(super) trait Compose: Sized {
    fn parenthesized(self) -> Self;
    fn negate(self) -> Result<Self, Self>;
    fn power(self, exponent: u32) -> Result<Self, Self>;
    fn binary(operator: BinaryOperator, left: Self, right: Self) -> Result<Self, (Self, Self)>;
    fn stepwise(
        operator: StepwiseOperator,
        first: Self,
        rest: Vec<Self>,
    ) -> Result<Self, (Self, Vec<Self>)>;
}

impl Compose for Forbidden {
    fn parenthesized(self) -> Self {
        match self {}
    }
    fn negate(self) -> Result<Self, Self> {
        match self {}
    }
    fn power(self, _: u32) -> Result<Self, Self> {
        match self {}
    }
    fn binary(_: BinaryOperator, left: Self, _: Self) -> Result<Self, (Self, Self)> {
        match left {}
    }
    fn stepwise(_: StepwiseOperator, first: Self, _: Vec<Self>) -> Result<Self, (Self, Vec<Self>)> {
        match first {}
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
    pub(super) left: Option<ExpressionAst<MultistageFormula>>,
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

    pub(super) fn annotate(self, annotation: L::Annotation) -> Option<Self> {
        match self {
            Self::Embedded(inner, None) => Some(Self::Embedded(inner, Some(annotation))),
            _ => None,
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

    fn negate(self) -> Result<Self, Self> {
        let operand = match self.into_inner() {
            Ok(inner) => match inner.negate() {
                Ok(inner) => return Ok(Self::embed(inner)),
                Err(inner) => Self::embed(inner),
            },
            Err(outer) => outer,
        };
        permitted(Box::new(operand), Self::Negate).map_err(|operand| *operand)
    }

    fn power(self, exponent: u32) -> Result<Self, Self> {
        let base = match self.into_inner() {
            Ok(inner) => match inner.power(exponent) {
                Ok(inner) => return Ok(Self::embed(inner)),
                Err(inner) => Self::embed(inner),
            },
            Err(outer) => outer,
        };
        permitted(Box::new(base), |permission, base| {
            Self::Power(permission, base, exponent)
        })
        .map_err(|base| *base)
    }

    fn binary(operator: BinaryOperator, left: Self, right: Self) -> Result<Self, (Self, Self)> {
        let (left, right) = match (left, right) {
            (Self::Embedded(left, None), Self::Embedded(right, None)) => {
                match L::Inner::binary(operator, left, right) {
                    Ok(inner) => return Ok(Self::embed(inner)),
                    Err((left, right)) => (Self::embed(left), Self::embed(right)),
                }
            }
            operands => operands,
        };
        operator.construct(left, right)
    }

    fn stepwise(
        operator: StepwiseOperator,
        first: Self,
        rest: Vec<Self>,
    ) -> Result<Self, (Self, Vec<Self>)> {
        let is_inner = |argument: &Self| matches!(argument, Self::Embedded(_, None));
        let (first, rest) = if is_inner(&first) && rest.iter().all(is_inner) {
            let unwrap = |argument| match argument {
                Self::Embedded(inner, None) => inner,
                _ => unreachable!("every argument is an unannotated embedding"),
            };
            match L::Inner::stepwise(
                operator,
                unwrap(first),
                rest.into_iter().map(unwrap).collect(),
            ) {
                Ok(inner) => return Ok(Self::embed(inner)),
                Err((first, rest)) => (
                    Self::embed(first),
                    rest.into_iter().map(Self::embed).collect(),
                ),
            }
        } else {
            (first, rest)
        };
        permitted((Box::new(first), rest), |permission, (first, rest)| {
            Self::Stepwise(permission, operator, first, rest)
        })
        .map_err(|(first, rest)| (*first, rest))
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

    fn built<T, R>(result: Result<T, R>) -> T {
        result.unwrap_or_else(|_| panic!("expected the construction to be permitted"))
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
        assert!(loaded().annotate(slopes("u")).is_none());
        let sum = built(Node::binary(BinaryOperator::Add, loaded(), column("firm")));
        assert!(sum.annotate(slopes("u")).is_none());
    }

    #[test]
    fn binary_operators_combine_in_the_innermost_shared_layer() {
        let interaction = built(Node::binary(
            BinaryOperator::Interact,
            column("firm"),
            column("sector"),
        ));
        assert!(matches!(
            interaction,
            ExpressionAst::Embedded(ExpressionAst::Interact(_, _), None)
        ));

        let ExpressionAst::Add(_, operands) =
            built(Node::binary(BinaryOperator::Add, interaction, loaded()))
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
        let call = |rest| built(Node::stepwise(StepwiseOperator::Sw, column("firm"), rest));
        assert!(matches!(
            call(vec![column("year")]),
            ExpressionAst::Embedded(ExpressionAst::Stepwise(..), None)
        ));

        let ExpressionAst::Stepwise(_, _, first, rest) = call(vec![loaded()]) else {
            panic!("expected a stepwise call in the fixed-effect layer");
        };
        assert!(matches!(*first, ExpressionAst::Embedded(_, None)));
        assert!(matches!(
            rest.as_slice(),
            [ExpressionAst::Embedded(_, Some(_))]
        ));
    }

    #[test]
    fn operators_forbidden_in_inner_layers_are_built_in_the_layer_that_permits_them() {
        enum Envelope {}
        impl Layer for Envelope {
            type Atom = Forbidden;
            type Operators = FormulaOperations;
            type Inner = Node;
            type Annotation = Forbidden;
            type Stepwise = Permitted;
        }
        type Outer = ExpressionAst<Envelope>;
        let column = |name| Outer::embed(column(name));

        // Interaction is permitted two layers down, in the grouping layer.
        assert!(matches!(
            built(Outer::binary(
                BinaryOperator::Interact,
                column("firm"),
                column("year")
            )),
            ExpressionAst::Embedded(
                ExpressionAst::Embedded(ExpressionAst::Interact(_, _), None),
                None
            )
        ));

        // Subtraction and negation are forbidden below the envelope, which permits them.
        assert!(matches!(
            built(Outer::binary(
                BinaryOperator::Subtract,
                column("firm"),
                column("year")
            )),
            ExpressionAst::Subtract(_, _)
        ));
        assert!(matches!(
            built(column("firm").negate()),
            ExpressionAst::Negate(_, _)
        ));
    }
}
