/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FINITEDIFFERENCELINEARIZATION_H
#define OPENKALMAN_FINITEDIFFERENCELINEARIZATION_H

#include <tuple>

namespace OpenKalman
{
  template<typename Trans, typename InDelta, typename ... PsDelta>
  struct FiniteDifferenceLinearization
  {
    static_assert(is_column_vector_v<InDelta>);
    static_assert((is_column_vector_v<PsDelta> and ...));
    static_assert(MatrixTraits<InDelta>::columns == 1);
    static_assert(((MatrixTraits<PsDelta>::columns == 1) and ...));
    static_assert(std::is_invocable_v<Trans, std::decay_t<InDelta>, std::decay_t<PsDelta>...>);

    FiniteDifferenceLinearization(Trans&& trans, InDelta&& in_delta, PsDelta&& ... ps_delta)
    : transformation(std::forward<Trans>(trans)),
      deltas(std::forward<InDelta>(in_delta), std::forward<PsDelta>(ps_delta)...) {}

  protected:
    template<typename In, typename ... Perturbations>
    static constexpr void check_inputs(In&&, Perturbations&& ...)
    {
      static_assert(is_equivalent_v<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>);
      static_assert(is_equivalent_v<typename MatrixTraits<In>::ColumnCoefficients, typename MatrixTraits<InDelta>::ColumnCoefficients>);
      static_assert((is_perturbation_v<Perturbations> and ...));
    }

    template<std::size_t term, typename...Inputs>
    auto jac_term(const std::tuple<Inputs...>& inputs) const
    {
      using Scalar = typename MatrixTraits<decltype(std::get<term>(inputs))>::Scalar;
      constexpr auto width = MatrixTraits<decltype(std::get<term>(inputs))>::dimension;
      return apply_columnwise<width>([&](std::size_t i) {
        const Scalar h = std::get<term>(deltas)[i];
        const auto t_start = internal::tuple_slice<0, term>(inputs);
        const auto t_end = internal::tuple_slice<1, sizeof...(Inputs)>(inputs);

        auto col = std::get<term>(inputs);
        const Scalar x = col[i];
        col[i] = x - h;
        const auto x1 = std::tuple_cat(t_start, std::tuple {col}, t_end);
        col[i] = x + h;
        const auto x2 = std::tuple_cat(t_start, std::tuple {col}, t_end);

        auto ret = strict((std::apply(transformation, x2) - std::apply(transformation, x1))/(2*h));
        return ret;
      });
    }

    template<typename...Inputs, std::size_t...ints>
    auto jacobian_impl(const std::tuple<Inputs...>& inputs, std::index_sequence<ints...>) const
    {
      static_assert(sizeof...(Inputs) == sizeof...(ints));
      return std::tuple {jac_term<ints>(inputs) ...};
    }

  public:
    /// Applies the transformation.
    template<typename In, typename ... Perturbations>
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      check_inputs(in, ps...);
      return transformation(in, ps...);
    }

    /// Returns a tuple of the Jacobians for the input and each perturbation term.
    template<typename In, typename ... Perturbations>
    auto jacobian(const In& in, const Perturbations&...ps) const
    {
      check_inputs(in, ps...);
      return jacobian_impl(std::tuple {in, ps...}, std::make_index_sequence<1 + sizeof...(ps)>());
    }

  protected:
    Trans transformation;
    const std::tuple<InDelta, PsDelta...> deltas;
  };


  /**
   * Deduction guide
   */

  template<typename Trans, typename InDelta, typename ... PsDelta>
  FiniteDifferenceLinearization(Trans&&, InDelta&&, PsDelta&&...)
    -> FiniteDifferenceLinearization<Trans, InDelta, PsDelta...>;


}


#endif //OPENKALMAN_FINITEDIFFERENCELINEARIZATION_H
