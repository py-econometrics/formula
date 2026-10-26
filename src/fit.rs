pub(crate) trait Definition<S> {
    type Input: ?Sized;
    type Output;
    type Error;

    fn fit(&self, input: &Self::Input) -> Result<S, Self::Error>;

    fn validate(&self, state: &S, input: &Self::Input) -> Result<(), Self::Error>;

    fn transform(&self, state: &S, input: &Self::Input) -> Result<Self::Output, Self::Error>;
}

impl<D: Definition<S>, S> Fitted<D, S> {
    pub(crate) fn fit(definition: D, input: &D::Input) -> Result<Self, D::Error> {
        let state = definition.fit(input)?;
        Ok(Self { definition, state })
    }

    pub(crate) fn transform(&self, input: &D::Input) -> Result<D::Output, D::Error> {
        self.definition.validate(&self.state, input)?;
        self.definition.transform(&self.state, input)
    }
}

impl<D: Definition<S> + Clone, S> Fit<D, S> {
    pub(crate) fn fit(&self, input: &D::Input) -> Result<Self, D::Error> {
        let fitted = Fitted::fit(self.definition().clone(), input)?;
        Ok(Self::Fitted(fitted))
    }
}

impl<D: Definition<S>, S> Fit<D, S> {
    pub(crate) fn transform(&self, input: &D::Input) -> Result<D::Output, FitError<D::Error>> {
        match self {
            Self::Unfitted(_) => Err(FitError::NotFitted),
            Self::Fitted(fitted) => fitted.transform(input).map_err(FitError::Definition),
        }
    }
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
