import polars

import formula

data = polars.DataFrame(
    {
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x1": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "x2": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "category": ["a", "b", "a", "b", "a", "b"],
        "firm": ["firm_a", "firm_b", "firm_c", "firm_a", "firm_b", "firm_c"],
        "year": [2020, 2021, 2022, 2020, 2021, 2022],
    }
)

# Parsing formula strings returns a collection of `Formula` objects
source = "y ~ sw(log(x1)*sqrt(x2), C(category, reference='a')) | firm + year"
formulas = formula.parse(source)  # Parse and expand formulas
first_formula = formulas[0]  # Parsed formulas are an indexable collection
same_formula = formulas[str(first_formula)]  # Look-up by formula-string also work

# A `Formula` object can "materialize" data according to the instructions of a formula string
fits = formulas.fit(data)  # Construct columns and learn transformation state
fit = fits[0]  # Integer indexing follows expansion order
same_fit = fits[fit.formula]  # Look up a fitted formula by its formula string

# A `FittedFormula` exposes its data alongside its meaning
frame = fit.frame()

# Select specific `Role`s, e.g., dependent variables, independent variables, and fixed effects
dependent = fit.select(formula.Role.Dependent).frame()
independent = fit.select(formula.Role.Independent).frame()
fixed_effects = fit.select(formula.Role.FixedEffect).frame()

# Select terms by their formula strings
interaction_terms = fit.select_terms("log(x1)*sqrt(x2)", role=formula.Role.Independent)
# Select the categorical term in the second formula
categorical_terms = fits["y ~ C(category, reference='a') | firm + year"].select_terms(
    "C(category, reference='a')", role=formula.Role.Independent
)

# Select roles before transforming data that has no dependent column.
predictors = fits.select(roles=[formula.Role.Independent, formula.Role.FixedEffect])
new_data = polars.DataFrame(
    {
        "x1": [8.0],
        "x2": [9.0],
        "category": ["b"],
        "firm": ["firm_a"],
        "year": [2020],
    }
)
transformed = predictors.transform(new_data)
