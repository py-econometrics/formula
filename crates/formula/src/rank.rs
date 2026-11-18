//! Structural full-rank coding of categorical factors, following Formulaic.
//!
//! A term is a set of factors. Its columns are described in the *reduced subterm*
//! basis: a term with categorical factors `F` and other factors `N` spans the
//! subterms `N ∪ S` for every `S ⊆ F`, each with every factor of `S` coded reduced
//! (treatment contrasts). Coding a factor *full* (one indicator per level) instead
//! spans a subterm both with and without that factor.
//!
//! Terms are taken in ascending degree. Each term emits only the subterms that the
//! fixed effects, the intercept, and earlier terms do not span yet, then marks them
//! spanned. The emitted subterms are compacted into column groups, merging `S` and
//! `S ∪ {f}` into `S ∪ {f}` with `f` full. A term may so emit several column groups,
//! each of which omits some of its factors, or none at all when it is fully spanned.
//!
//! Explicitly omitted levels restrict a factor to literal indicators. For axes
//! with such restrictions, we track sparse coordinates in a basis of the constant
//! and all but one level indicator. Unrestricted axes keep their scoped subterms,
//! so their levels need not be enumerated to decide structural rank.

use std::cmp::Reverse;
use std::collections::BTreeMap;

use num_bigint::{BigInt, Sign};
use num_integer::Integer;

/// One factor of a term, identified by its typed input key.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RankFactor<'a, K> {
    pub(crate) key: K,
    pub(crate) levels: Option<usize>,
    pub(crate) omitted: &'a [usize],
}

/// How one column group of a term materialises one of the term's factors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    /// The column group does not include the factor.
    Omitted,
    /// Every level but the reference (categorical factors only).
    Reduced,
    /// Every level of a categorical factor, or the factor's own columns.
    Full,
    /// Selected fitted level indices, in fitted order.
    Levels(Vec<usize>),
}

/// One column group of a term: a scope for each factor of the term, in term order.
pub(crate) type ColumnGroup = Vec<Scope>;

/// A set of factor keys, without duplicates.
type KeySet<K> = Vec<K>;

fn same_set<K: PartialEq>(left: &[K], right: &[K]) -> bool {
    left.len() == right.len() && left.iter().all(|key| right.contains(key))
}

fn contains<K: PartialEq>(spanned: &[KeySet<K>], set: &[K]) -> bool {
    spanned.iter().any(|other| same_set(other, set))
}

/// One axis of the fitted term. Repeated aliases share an axis and combine
/// their omissions. Categorical aliases materialize at the first position.
struct Axis<K> {
    key: K,
    levels: Option<usize>,
    omitted: Vec<usize>,
    positions: Vec<usize>,
}

fn axes<K: PartialEq + Copy>(term: &[RankFactor<'_, K>]) -> Vec<Axis<K>> {
    let mut axes: Vec<Axis<K>> = Vec::new();
    for (position, factor) in term.iter().enumerate() {
        if let Some(axis) = axes.iter_mut().find(|axis| axis.key == factor.key) {
            debug_assert_eq!(axis.levels, factor.levels);
            axis.omitted.extend_from_slice(factor.omitted);
            if axis.levels.is_none() {
                axis.positions.push(position);
            }
        } else {
            axes.push(Axis {
                key: factor.key,
                levels: factor.levels,
                omitted: factor.omitted.to_vec(),
                positions: vec![position],
            });
        }
    }
    for axis in &mut axes {
        axis.omitted.sort_unstable();
        axis.omitted.dedup();
    }
    axes
}

fn included(group: &[Scope]) -> usize {
    group
        .iter()
        .filter(|scope| **scope != Scope::Omitted)
        .count()
}

/// The reduced subterms a term spans, in Formulaic's order.
///
/// That is the Cartesian product over the term's keys, the first key varying
/// slowest, of (reduced, omitted) for categorical keys and (full) for the others,
/// stably sorted by the number of included keys.
fn subterms(keys: &[(impl Copy, bool)]) -> Vec<ColumnGroup> {
    let mut subterms = vec![ColumnGroup::new()];
    for &(_, categorical) in keys {
        let options: &[Scope] = if categorical {
            &[Scope::Reduced, Scope::Omitted]
        } else {
            &[Scope::Full]
        };
        subterms = subterms
            .iter()
            .flat_map(|prefix| {
                options.iter().map(move |scope| {
                    let mut subterm = prefix.clone();
                    subterm.push(scope.clone());
                    subterm
                })
            })
            .collect();
    }
    subterms.sort_by_key(|subterm| included(subterm));
    subterms
}

/// The minimal column groups spanning the same space as `groups`, in Formulaic's order.
///
/// Port of Formulaic's `_simplify_scoped_terms`: a group that adds one reduced
/// factor to an existing group replaces it, with that factor full, and the result
/// is simplified again.
fn simplify(mut groups: Vec<ColumnGroup>) -> Vec<ColumnGroup> {
    groups.sort_by_key(|group| included(group));
    let mut simplified: Vec<ColumnGroup> = Vec::new();
    for group in groups {
        let merge = simplified.iter().enumerate().find_map(|(index, existing)| {
            if included(&group) != included(existing) + 1 {
                return None;
            }
            let mut added = (0..group.len())
                .filter(|&position| group[position] != Scope::Omitted)
                .filter(|&position| group[position] != existing[position]);
            match (added.next(), added.next()) {
                (Some(position), None) if group[position] == Scope::Reduced => {
                    Some((index, position))
                }
                _ => None,
            }
        });
        match merge {
            Some((index, position)) => {
                let mut merged = group;
                merged[position] = Scope::Full;
                simplified.remove(index);
                if !simplified.contains(&merged) {
                    simplified.push(merged);
                }
                simplified = simplify(simplified);
            }
            None => {
                if !simplified.contains(&group) {
                    simplified.push(group);
                }
            }
        }
    }
    simplified
}

/// An affected axis and its fitted reference for symbolic coordinates.
/// The first explicit omission is a useful reference because all the retained
/// indicators of that restriction then have single coordinates.
struct LevelAxis<K> {
    key: K,
    reference: usize,
}

/// Each coordinate lists only the affected axes with a non-reference indicator.
/// Larger supports pivot before smaller ones. The constant has empty support.
type Coordinate = (Reverse<usize>, Vec<(usize, usize)>);
type CoefficientRow = BTreeMap<Coordinate, BigInt>;

#[derive(Default)]
struct Basis {
    pivots: BTreeMap<Coordinate, CoefficientRow>,
}

impl Basis {
    fn insert(&mut self, mut row: CoefficientRow) -> bool {
        while let Some(pivot) = row.keys().next().cloned() {
            let Some(existing) = self.pivots.get(&pivot) else {
                self.pivots.insert(pivot, row);
                return true;
            };
            // A single-coordinate pivot row eliminates its coordinate outright.
            if existing.len() == 1 {
                row.remove(&pivot);
                continue;
            }

            // Cross-multiply to eliminate the pivot with integer coefficients.
            // Removing common divisors keeps the intermediate rows small.
            let value = &row[&pivot];
            let previous = &existing[&pivot];
            let divisor = previous.gcd(value);
            let left = previous / &divisor;
            let right = value / &divisor;
            if left != BigInt::ONE {
                for entry in row.values_mut() {
                    *entry *= &left;
                }
            }
            for (coordinate, coefficient) in existing {
                *row.entry(coordinate.clone()).or_default() -= coefficient * &right;
            }
            row.retain(|_, coefficient| coefficient.sign() != Sign::NoSign);
            let divisor = row
                .values()
                .try_fold(BigInt::ZERO, |divisor, value| {
                    let divisor = divisor.gcd(value);
                    if divisor == BigInt::ONE {
                        None
                    } else {
                        Some(divisor)
                    }
                })
                .unwrap_or(BigInt::ONE);
            if divisor > BigInt::ONE {
                for entry in row.values_mut() {
                    *entry /= &divisor;
                }
            }
        }
        false
    }
}

/// A candidate indicator product in the affected axes' symbolic coordinates.
/// Unaffected axes are constant or reduced throughout the surrounding block.
fn coordinates<K: Copy>(
    chosen: &[(usize, usize, usize)],
    level_axes: &[LevelAxis<K>],
    block: &[K],
    covered_by_fe: &impl Fn(&[K]) -> bool,
) -> CoefficientRow {
    let mut row = vec![(Vec::new(), 1)];
    for &(axis, levels, level) in chosen {
        let reference = level_axes[axis].reference;
        if level == reference {
            let mut expanded = Vec::new();
            for (support, coefficient) in &row {
                for other in (0..levels).filter(|&other| other != reference) {
                    let mut support = support.clone();
                    support.push((axis, other));
                    expanded.push((support, -coefficient));
                }
            }
            row.extend(expanded);
        } else {
            for (support, _) in &mut row {
                support.push((axis, level));
            }
        }
    }
    row.into_iter()
        .filter_map(|(mut support, coefficient)| {
            let mut keys = block.to_vec();
            keys.extend(support.iter().map(|&(axis, _)| level_axes[axis].key));
            if covered_by_fe(&keys) {
                return None;
            }
            support.sort_unstable();
            Some(((Reverse(support.len()), support), BigInt::from(coefficient)))
        })
        .collect()
}

/// Accept the independent indicator products for one unaffected subterm. Walk
/// levels in fitted order, with the default reference last for unrestricted
/// factors, and combine adjacent products with the same suffix scopes.
fn select_levels<K: Copy>(
    axes: &[(usize, &Axis<K>, usize)],
    chosen: &mut Vec<(usize, usize, usize)>,
    accept: &mut impl FnMut(&[(usize, usize, usize)]) -> bool,
) -> Vec<Vec<Scope>> {
    let Some((&(id, axis, levels), tail)) = axes.split_first() else {
        return if accept(chosen) {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    };
    let mut selections = Vec::new();
    let default_reference_last = axis.omitted.is_empty() && levels > 1;
    let order = if default_reference_last {
        1..levels
    } else {
        0..levels
    };
    for level in order
        .chain(default_reference_last.then_some(0))
        .filter(|&level| level < levels && axis.omitted.binary_search(&level).is_err())
    {
        chosen.push((id, levels, level));
        let suffixes = select_levels(tail, chosen, accept);
        chosen.pop();
        if !suffixes.is_empty() {
            selections.push((level, suffixes));
        }
    }
    selections.sort_unstable_by_key(|(level, _)| *level);
    let mut groups: Vec<(Vec<usize>, Vec<Vec<Scope>>)> = Vec::new();
    for (level, suffixes) in selections {
        if let Some((previous_levels, previous_suffixes)) = groups.last_mut()
            && suffixes.len() == 1
            && *previous_suffixes == suffixes
        {
            previous_levels.push(level);
        } else {
            groups.push((vec![level], suffixes));
        }
    }
    groups
        .into_iter()
        .flat_map(|(selected, suffixes)| {
            let scope = if selected.len() == levels {
                Scope::Full
            } else if axis.omitted.is_empty() && selected.len() + 1 == levels && selected[0] == 1 {
                Scope::Reduced
            } else {
                Scope::Levels(selected)
            };
            suffixes.into_iter().map(move |mut suffix| {
                suffix.insert(0, scope.clone());
                suffix
            })
        })
        .collect()
}

/// Choose groups in stable ascending degree order and return them in written
/// order. Fixed-effect coverage is queried only for coordinates reached by an
/// actual term.
pub(crate) fn code_terms<K: PartialEq + Copy>(
    terms: &[Vec<RankFactor<'_, K>>],
    covered_by_fe: impl Fn(&[K]) -> bool,
) -> Vec<Vec<ColumnGroup>> {
    let terms_axes = terms.iter().map(|term| axes(term)).collect::<Vec<_>>();
    let mut order = (0..terms.len()).collect::<Vec<_>>();
    order.sort_by_key(|&index| terms_axes[index].len());

    let mut level_axes: Vec<LevelAxis<K>> = Vec::new();
    for axis in terms_axes.iter().flatten() {
        if let Some(&reference) = axis.omitted.first()
            && Some(axis.omitted.len()) != axis.levels
            && !level_axes.iter().any(|entry| entry.key == axis.key)
        {
            level_axes.push(LevelAxis {
                key: axis.key,
                reference,
            });
        }
    }

    let mut spanned = Vec::new();
    let mut bases: Vec<(KeySet<K>, Basis)> = Vec::new();
    let mut coded = vec![Vec::new(); terms.len()];
    for index in order {
        let axes = &terms_axes[index];
        if axes
            .iter()
            .any(|axis| !axis.omitted.is_empty() && Some(axis.omitted.len()) == axis.levels)
        {
            let group = axes
                .iter()
                .map(|axis| {
                    if axis.omitted.is_empty() {
                        Scope::Full
                    } else {
                        Scope::Levels(
                            (0..axis.levels.expect("only categorical factors omit levels"))
                                .filter(|level| axis.omitted.binary_search(level).is_err())
                                .collect(),
                        )
                    }
                })
                .collect();
            coded[index].push(expand_group(axes, group, terms[index].len()));
            continue;
        }

        let affected = axes
            .iter()
            .enumerate()
            .filter_map(|(position, axis)| {
                let id = level_axes.iter().position(|entry| entry.key == axis.key)?;
                Some((
                    position,
                    id,
                    axis,
                    axis.levels.expect("affected axes are categorical"),
                ))
            })
            .collect::<Vec<_>>();
        let unaffected = axes
            .iter()
            .enumerate()
            .filter(|(position, _)| {
                !affected
                    .iter()
                    .any(|(affected_position, ..)| affected_position == position)
            })
            .collect::<Vec<_>>();
        let keys = unaffected
            .iter()
            .map(|(_, axis)| (axis.key, axis.levels.is_some()))
            .collect::<Vec<_>>();
        let affected_axes = affected
            .iter()
            .map(|&(_, id, axis, levels)| (id, axis, levels))
            .collect::<Vec<_>>();

        let mut emitted = Vec::new();
        for subterm in subterms(&keys) {
            let block = keys
                .iter()
                .zip(&subterm)
                .filter(|(_, scope)| **scope != Scope::Omitted)
                .map(|((key, _), _)| *key)
                .collect::<Vec<_>>();
            // A covered block can still have uncovered affected contrasts.
            if affected.is_empty() && covered_by_fe(&block) {
                continue;
            }
            if level_axes.is_empty() {
                if !contains(&spanned, &block) {
                    spanned.push(block);
                    emitted.push(subterm);
                }
                continue;
            }
            let basis = match bases.iter().position(|(keys, _)| same_set(keys, &block)) {
                Some(position) => &mut bases[position].1,
                None => {
                    bases.push((block.clone(), Basis::default()));
                    &mut bases.last_mut().expect("just inserted a basis").1
                }
            };
            for selection in select_levels(&affected_axes, &mut Vec::new(), &mut |chosen| {
                basis.insert(coordinates(chosen, &level_axes, &block, &covered_by_fe))
            }) {
                let mut group = vec![Scope::Omitted; axes.len()];
                for (&(position, _), scope) in unaffected.iter().zip(&subterm) {
                    group[position] = scope.clone();
                }
                for (&(position, ..), scope) in affected.iter().zip(selection) {
                    group[position] = scope;
                }
                emitted.push(group);
            }
        }

        coded[index] = simplify(emitted)
            .into_iter()
            .map(|group| expand_group(axes, group, terms[index].len()))
            .collect();
    }
    coded
}

fn expand_group<K>(axes: &[Axis<K>], group: ColumnGroup, width: usize) -> ColumnGroup {
    let mut expanded = vec![Scope::Omitted; width];
    for (axis, scope) in axes.iter().zip(group) {
        for &position in &axis.positions {
            expanded[position] = scope.clone();
        }
    }
    expanded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(spec: &str) -> Vec<RankFactor<'static, char>> {
        spec.chars()
            .map(|key| RankFactor {
                key,
                levels: (key != 'x').then_some(4),
                omitted: &[],
            })
            .collect()
    }

    /// Renders a column group like `a-:b`, with `-` for reduced and `1` for none.
    fn render(spec: &str, group: &[Scope]) -> String {
        let factors = spec
            .chars()
            .zip(group)
            .filter_map(|(key, scope)| match scope {
                Scope::Omitted => None,
                Scope::Reduced => Some(format!("{key}-")),
                Scope::Full => Some(key.to_string()),
                Scope::Levels(levels) => Some(format!("{key}{levels:?}")),
            })
            .collect::<Vec<_>>();
        if factors.is_empty() {
            "1".to_owned()
        } else {
            factors.join(":")
        }
    }

    fn code_seeded(seeds: Vec<Vec<char>>, terms: &[&str]) -> Vec<Vec<String>> {
        let factors = terms.iter().map(|spec| term(spec)).collect::<Vec<_>>();
        code_terms(&factors, |keys| contains(&seeds, keys))
            .into_iter()
            .zip(terms)
            .map(|(groups, spec)| groups.iter().map(|group| render(spec, group)).collect())
            .collect()
    }

    fn code(terms: &[&str]) -> Vec<Vec<String>> {
        code_seeded(Vec::new(), terms)
    }

    #[test]
    fn main_effects_reduce_after_the_constant_is_spanned() {
        assert_eq!(code(&["", "a", "b"]), [["1"], ["a-"], ["b-"]]);
        assert_eq!(code(&["a", "b"]), [["a"], ["b-"]]);
    }

    #[test]
    fn terms_are_decided_by_degree_and_reported_in_order() {
        assert_eq!(
            code(&["", "ab", "a", "b"]),
            [["1"], ["a-:b-"], ["a-"], ["b-"]]
        );
    }

    #[test]
    fn numeric_factors_take_part_in_the_span() {
        assert_eq!(code(&["x", "ax"]), [vec!["x"], vec!["a-:x"]]);
        assert_eq!(code(&["", "ax"]), [vec!["1"], vec!["a:x"]]);
    }

    #[test]
    fn a_term_emits_several_column_groups_where_one_coding_cannot_suffice() {
        assert_eq!(code(&["", "ab"]), [vec!["1"], vec!["b-", "a-:b"]]);
        assert_eq!(code(&["a", "bc"]), [vec!["a"], vec!["c-", "b-:c"]]);
        assert_eq!(code(&["", "abc"])[1], ["a-:b", "a:c-", "b-:c", "a-:b-:c-"]);
        assert_eq!(code(&["", "ab", "abc"])[2], ["a:b:c-"]);
    }

    #[test]
    fn fully_spanned_terms_emit_no_column_groups() {
        assert_eq!(code(&["", "a", "a"])[2], Vec::<String>::new());
        assert_eq!(code(&["", "x", "x"])[2], Vec::<String>::new());
        assert_eq!(code(&["", "ab", "ba"])[2], Vec::<String>::new());
    }

    #[test]
    fn seeds_are_spanned_before_any_term() {
        let seeds = vec![vec![], vec!['f'], vec!['f'], vec!['x'], vec!['x', 'f']];
        assert_eq!(
            code_seeded(seeds, &["a", "x", "af", "fx"]),
            [vec!["a-"], vec![], vec!["a-:f-"], vec![]]
        );
        // Without the main effect, `a` is reduced within the levels of `f`.
        let seeds = vec![vec![], vec!['f']];
        assert_eq!(code_seeded(seeds, &["af"]), [["a-:f"]]);
    }

    #[test]
    fn repeated_categorical_keys_materialize_once() {
        let mut factors = term("a");
        factors.push(RankFactor {
            key: 'a',
            levels: Some(4),
            omitted: &[],
        });
        assert_eq!(
            code_terms(&[factors], |_| false),
            [vec![vec![Scope::Full, Scope::Omitted]]]
        );
    }

    fn restricted(key: char, omitted: &[usize]) -> RankFactor<'_, char> {
        RankFactor {
            key,
            levels: Some(4),
            omitted,
        }
    }

    fn code_with_omissions(terms: &[Vec<RankFactor<'_, char>>]) -> Vec<Vec<ColumnGroup>> {
        code_terms(terms, |_| false)
    }

    #[test]
    fn repeated_aliases_intersect_kept_levels() {
        let terms = [vec![restricted('a', &[0]), restricted('a', &[1])]];
        assert_eq!(
            code_with_omissions(&terms),
            [vec![vec![Scope::Levels(vec![2, 3]), Scope::Omitted]]],
        );
    }

    #[test]
    fn all_level_omission_does_not_cover_a_subsequent_term() {
        let terms = [vec![restricted('a', &[0, 0, 1, 2, 3])], term("a")];
        assert_eq!(
            code_with_omissions(&terms),
            [vec![vec![Scope::Levels(vec![])]], vec![vec![Scope::Full]]],
        );
    }

    #[test]
    fn partial_overlaps_add_only_previously_uncovered_levels() {
        for (first, next, kept_first, kept_next) in [
            (&[0, 1][..], &[1, 2][..], vec![2, 3], vec![0]),
            (&[1, 2][..], &[0, 1][..], vec![0, 3], vec![2]),
        ] {
            for intercept in [false, true] {
                let mut terms = vec![vec![restricted('a', first)], vec![restricted('a', next)]];
                if intercept {
                    terms.insert(0, vec![]);
                }
                assert_eq!(
                    code_with_omissions(&terms)[usize::from(intercept)..],
                    [
                        vec![vec![Scope::Levels(kept_first.clone())]],
                        vec![vec![Scope::Levels(kept_next.clone())]],
                    ],
                );
            }
        }
    }

    #[test]
    fn restricted_and_unrestricted_terms_share_their_actual_span() {
        let restricted = vec![restricted('a', &[0, 1])];
        assert_eq!(
            code_with_omissions(&[vec![], restricted.clone(), term("a")]),
            [
                vec![vec![]],
                vec![vec![Scope::Levels(vec![2, 3])]],
                vec![vec![Scope::Levels(vec![1])]]
            ],
        );
        assert_eq!(
            code_with_omissions(&[vec![], term("a"), restricted]),
            [vec![vec![]], vec![vec![Scope::Reduced]], vec![]],
        );
    }

    #[test]
    fn restricted_interactions_share_coverage_in_each_unrestricted_block() {
        let terms = [
            term("b"),
            vec![restricted('a', &[0, 1]), term("b")[0]],
            vec![term("b")[0], restricted('a', &[1, 2])],
        ];
        assert_eq!(
            code_with_omissions(&terms),
            [
                vec![vec![Scope::Full]],
                vec![vec![Scope::Levels(vec![2, 3]), Scope::Full]],
                vec![vec![Scope::Full, Scope::Levels(vec![0])]],
            ],
        );
    }

    #[test]
    fn unaffected_high_cardinality_does_not_expand_the_level_grid() {
        let large = RankFactor {
            key: 'b',
            levels: Some(1_000_000),
            omitted: &[],
        };
        let terms = [vec![restricted('a', &[0, 1]), large]];
        let queries = std::cell::Cell::new(0);
        let coding = code_terms(&terms, |_| {
            queries.set(queries.get() + 1);
            false
        });
        assert_eq!(coding, [vec![vec![Scope::Levels(vec![2, 3]), Scope::Full]]]);
        assert_eq!(queries.get(), 4);
    }

    #[test]
    fn fe_coverage_can_absorb_one_axis_of_a_restricted_interaction() {
        let terms = [vec![restricted('a', &[0, 1]), term("b")[0]]];
        let coding = code_terms(&terms, |keys| keys.iter().all(|key| *key == 'b'));
        assert_eq!(coding, [vec![vec![Scope::Levels(vec![2, 3]), Scope::Full]]]);
        let coding = code_terms(&terms, |keys| keys.iter().all(|key| *key == 'a'));
        assert_eq!(
            coding,
            [vec![vec![Scope::Levels(vec![2, 3]), Scope::Reduced]]]
        );
        let coding = code_terms(&terms, |_| true);
        assert_eq!(coding, [Vec::<ColumnGroup>::new()]);
    }
}
