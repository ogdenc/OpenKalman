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
    static_assert(not is_wrapped_v<std::invoke_result_t<Trans, std::decay_t<InDelta>, std::decay_t<PsDelta>...>>,
      "For finite difference linearization, the transformation function cannot return a wrapped matrix.");

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

    template<typename T>
    static constexpr auto make_strict_typed_matrix(T&& t)
    {
      using RC = typename MatrixTraits<T>::RowCoefficients;
      using CC = typename MatrixTraits<T>::ColumnCoefficients;
      return TypedMatrix<RC, CC, strict_matrix_t<T>> {std::move(t)};
    }

    template<std::size_t term, typename...Inputs>
    auto jac_term(std::tuple<Inputs...>&& inputs) const
    {
      using Traits = MatrixTraits<decltype(std::get<term>(inputs))>;
      using Scalar = typename Traits::Scalar;
      constexpr auto width = Traits::dimension;
      return apply_columnwise<width>([&](std::size_t i) {
        const Scalar h = std::get<term>(deltas)[i];
        auto& col = std::get<term>(inputs);
        const Scalar x = col[i];
        col[i] = x + h;
        const auto fp = std::apply(transformation, inputs);
        col[i] = x - h;
        const auto fm = std::apply(transformation, inputs);
        col[i] = x;
        return strict((fp - fm)/(2*h));
      });
    }

    template<typename...Inputs, std::size_t...terms>
    auto jacobian_impl(std::tuple<Inputs...>&& inputs, std::index_sequence<terms...>) const
    {
      static_assert(sizeof...(Inputs) == sizeof...(terms));
      return std::tuple {jac_term<terms>(std::move(inputs)) ...};
    }

    template<std::size_t term, std::size_t i, std::size_t j, typename...Inputs>
    auto h_j(std::tuple<Inputs...>&& inputs) const
    {
      using Scalar = typename MatrixTraits<decltype(std::get<term>(inputs))>::Scalar;
      const Scalar hi = std::get<term>(deltas)[i];
      auto& col = std::get<term>(inputs);
      const Scalar xi = col[i];

      if constexpr (i == j)
      {
        const auto f0 = std::apply(transformation, inputs);
        col[i] = xi + hi;
        const auto fp = std::apply(transformation, inputs);
        col[i] = xi - hi;
        const auto fm = std::apply(transformation, inputs);
        col[i] = xi;
        auto ret = strict((fp - 2*f0 + fm) / (hi * hi));
        return ret;
      }
      else
      {
        const Scalar hj = std::get<term>(deltas)[j];
        const Scalar xj = col[j];
        col[i] = xi + hi;
        col[j] = xj + hj;
        const auto fpp = std::apply(transformation, inputs);
        col[j] = xj - hj;
        const auto fpm = std::apply(transformation, inputs);
        col[i] = xi - hi;
        const auto fmm = std::apply(transformation, inputs);
        col[j] = xj + hj;
        const auto fmp = std::apply(transformation, inputs);
        col[i] = xi;
        col[j] = xj;
        auto ret = strict((fpp - fpm - fmp + fmm) / (4 * hi * hj));
        return ret;
      }
    };

    template<std::size_t term, std::size_t i, typename...Inputs, std::size_t...js>
    auto h_i(std::tuple<Inputs...>&& inputs, std::index_sequence<js...>) const
    {
      return std::array {h_j<term, i, js>(std::move(inputs))...};
    }

    template<std::size_t term, typename...Inputs, std::size_t...is>
    auto h_k(std::tuple<Inputs...>&& inputs, std::index_sequence<is...>) const
    {
      constexpr auto j_size = MatrixTraits<decltype(std::get<term>(inputs))>::dimension;
      using A = decltype(h_i<term, 0>(std::move(inputs), std::make_index_sequence<j_size>()));
      return std::array<A, sizeof...(is)> {h_i<term, is>(std::move(inputs), std::make_index_sequence<j_size>())...};
    }

    template<std::size_t term, typename...Inputs, std::size_t...ks>
    auto h_term(std::tuple<Inputs...>&& inputs, std::index_sequence<ks...>) const
    {
      using TermTrait = MatrixTraits<decltype(std::get<term>(inputs))>;
      constexpr auto i_size = TermTrait::dimension;
      const auto t = h_k<term>(std::move(inputs), std::make_index_sequence<i_size>());
      constexpr auto width = TermTrait::dimension;
      using C = typename TermTrait::RowCoefficients;
      using Vb = typename TermTrait::template StrictMatrix<width, width>;
      using V = TypedMatrix<C, C, Vb>;
      return std::array {apply_coefficientwise<V>([&](std::size_t i, std::size_t j) { return t[i][j][ks]; })...};
    }

    template<typename...Inputs, std::size_t...terms>
    auto hessian_impl(std::tuple<Inputs...>&& inputs, std::index_sequence<terms...>) const
    {
      static_assert(sizeof...(Inputs) == sizeof...(terms));
      constexpr auto k_size = MatrixTraits<std::invoke_result_t<Trans, Inputs...>>::dimension;
      return std::tuple {h_term<terms>(std::move(inputs), std::make_index_sequence<k_size>())...};
    }

  public:
    /// Applies the transformation.
    template<typename In, typename ... Perturbations>
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      check_inputs(in, ps...);
      return transformation(std::forward<In>(in), std::forward<Perturbations>(ps)...);
    }

    /// Returns a tuple of the Jacobians for the input and each perturbation term.
    template<typename In, typename ... Perturbations>
    auto jacobian(In&& in, Perturbations&&...ps) const
    {
      check_inputs(in, ps...);
      return jacobian_impl(
        std::forward_as_tuple(make_strict_typed_matrix(std::forward<In>(in)),
          make_strict_typed_matrix(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<1 + sizeof...(ps)>());
    }

    /// Returns a tuple of Hessian matrices for the input and each perturbation term.
    template<typename In, typename ... Perturbations>
    auto hessian(In&& in, Perturbations&&...ps) const
    {
      check_inputs(in, ps...);
      return hessian_impl(
        std::forward_as_tuple(make_strict_typed_matrix(std::forward<In>(in)),
          make_strict_typed_matrix(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<1 + sizeof...(ps)>());
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
