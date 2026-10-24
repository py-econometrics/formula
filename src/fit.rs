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
