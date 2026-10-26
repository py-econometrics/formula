pub(crate) trait Definition<S> {
    type Input: ?Sized;
    type Output;
    type Error;

    fn fit(&self, input: &Self::Input) -> Result<S, Self::Error>;

    fn validate(&self, state: &S, input: &Self::Input) -> Result<(), Self::Error>;

    fn transform(&self, state: &S, input: &Self::Input) -> Result<Self::Output, Self::Error>;
}

#[derive(Debug)]
pub enum FitError<E> {
    NotFitted,
    Definition(E),
}

#[derive(Debug)]
pub(crate) enum Fit<D, S> {
    Unfitted(D),
    Fitted(Fitted<D, S>),
}

#[derive(Debug)]
pub(crate) struct Fitted<D, S> {
    pub(crate) definition: D,
    pub(crate) state: S,
}

impl<D, S> Fitted<D, S> {
    pub(crate) fn new(definition: D, state: S) -> Self {
        Self { definition, state }
    }
}

impl<D, S> Fit<D, S> {
    pub(crate) fn definition(&self) -> &D {
        match self {
            Self::Unfitted(definition) => definition,
            Self::Fitted(fitted) => &fitted.definition,
        }
    }
}

// Fitted state does not affect structural identity.
impl<D: PartialEq, S> PartialEq for Fit<D, S> {
    fn eq(&self, other: &Self) -> bool {
        self.definition() == other.definition()
    }
}
