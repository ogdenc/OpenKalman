/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FINITEDIFFERENCELINEARIZATION_HPP
#define OPENKALMAN_FINITEDIFFERENCELINEARIZATION_HPP

#include <tuple>

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<typename Trans, transformation_input InDelta, transformation_input ... PsDelta> requires
    std::invocable<Trans, InDelta, PsDelta...> and
    (not wrapped_mean<std::invoke_result_t<Trans, InDelta, PsDelta...>>) and
    (not std::is_reference_v<InDelta>) and (((not std::is_reference_v<PsDelta>) and ...))
#else
  template<typename Trans, typename InDelta, typename ... PsDelta>
#endif
  struct FiniteDifferenceLinearization
  {
#ifndef __cpp_concepts
    static_assert((transformation_input<InDelta> and ... and transformation_input<PsDelta>));
    static_assert(std::is_invocable_v<Trans, InDelta, PsDelta...>);
    static_assert(not wrapped_mean<std::invoke_result_t<Trans, InDelta, PsDelta...>>,
      "For finite difference linearization, the transformation function cannot return a wrapped matrix.");
    static_assert(not std::is_reference_v<InDelta>);
    static_assert(((not std::is_reference_v<PsDelta>) and ...));
#endif

    FiniteDifferenceLinearization(Trans&& trans, InDelta&& in_delta, PsDelta&& ... ps_delta)
    : transformation(std::forward<Trans>(trans)),
      deltas(std::forward<InDelta>(in_delta), std::forward<PsDelta>(ps_delta)...) {}

  private:

    template<typename T>
    static constexpr auto make_strict_typed_matrix(T&& t)
    {
      using RC = typename MatrixTraits<T>::RowCoefficients;
      using CC = typename MatrixTraits<T>::ColumnCoefficients;
      return Matrix<RC, CC, native_matrix_t<T>> {std::move(t)};
    }


    template<std::size_t term, typename...Inputs>
    auto jac_term(std::tuple<Inputs...>&& inputs) const
    {
      using Traits = MatrixTraits<decltype(std::get<term>(inputs))>;
      using Scalar = typename Traits::Scalar;
      constexpr auto width = Traits::dimension;
      auto& col = std::get<term>(inputs);
      return apply_columnwise<width>([&](std::size_t i) {
        const Scalar h = std::get<term>(deltas)[i];
        const Scalar x = col[i];
        col[i] = x + h;
        const auto fp = Mean {std::apply(transformation, inputs)};
        col[i] = x - h;
        const auto fm = Mean {std::apply(transformation, inputs)};
        col[i] = x;
        return make_self_contained((fp - fm)/(2*h));
      });
    }


#ifdef __cpp_concepts
    template<typename...Inputs, std::size_t...terms> requires (sizeof...(Inputs) == sizeof...(terms))
#else
    template<typename...Inputs, std::size_t...terms, std::enable_if_t<sizeof...(Inputs) == sizeof...(terms), int> = 0>
#endif
    auto jacobian_impl(std::tuple<Inputs...>&& inputs, std::index_sequence<terms...>) const
    {
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
        const auto f0 = Mean {std::apply(transformation, inputs)};
        col[i] = xi + hi;
        const auto fp = Mean {std::apply(transformation, inputs)};
        col[i] = xi - hi;
        const auto fm = Mean {std::apply(transformation, inputs)};
        col[i] = xi;
        auto ret = make_self_contained(((fp - f0) - (f0 - fm)) / (hi * hi)); // Use two separate subtractions to ensure proper wrapping.
        return ret;
      }
      else
      {
        const Scalar hj = std::get<term>(deltas)[j];
        const Scalar xj = col[j];
        col[i] = xi + hi;
        col[j] = xj + hj;
        const auto fpp = Mean {std::apply(transformation, inputs)};
        col[j] = xj - hj;
        const auto fpm = Mean {std::apply(transformation, inputs)};
        col[i] = xi - hi;
        const auto fmm = Mean {std::apply(transformation, inputs)};
        col[j] = xj + hj;
        const auto fmp = Mean {std::apply(transformation, inputs)};
        col[i] = xi;
        col[j] = xj;
        auto ret = make_self_contained(((fpp - fpm) - (fmp - fmm)) / (4 * hi * hj));
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
      using Term = decltype(std::get<term>(inputs));
      using TermTrait = MatrixTraits<Term>;
      constexpr auto i_size = TermTrait::dimension;
      const auto t = h_k<term>(std::move(inputs), std::make_index_sequence<i_size>());
      constexpr auto width = TermTrait::dimension;
      using C = typename TermTrait::RowCoefficients;
      using Vb = native_matrix_t<Term, width, width>;
      using V = Matrix<C, C, Vb>;
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
#ifdef __cpp_concepts
    template<transformation_input In, perturbation ... Perturbations> requires
      equivalent_to<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations> and ...) and
      equivalent_to<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>,
        int> = 0>
#endif
    auto operator()(In&& in, Perturbations&& ... ps) const
    {
      return transformation(std::forward<In>(in), std::forward<Perturbations>(ps)...);
    }


    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation ... Perturbations> requires
      equivalent_to<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations> and ...) and
        equivalent_to<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>,
          int> = 0>
#endif
    auto jacobian(In&& in, Perturbations&&...ps) const
    {
      return jacobian_impl(
        std::forward_as_tuple(make_strict_typed_matrix(std::forward<In>(in)),
          make_strict_typed_matrix(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<1 + sizeof...(ps)>());
    }


    /// Returns a tuple of Hessian matrices for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation ... Perturbations> requires
      equivalent_to<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations> and ...) and
        equivalent_to<typename MatrixTraits<In>::RowCoefficients, typename MatrixTraits<InDelta>::RowCoefficients>,
          int> = 0>
#endif
    auto hessian(In&& in, Perturbations&&...ps) const
    {
      return hessian_impl(
        std::forward_as_tuple(make_strict_typed_matrix(std::forward<In>(in)),
          make_strict_typed_matrix(std::forward<Perturbations>(ps))...),
        std::make_index_sequence<1 + sizeof...(ps)>());
    }

  protected:

    Trans transformation;
    const std::tuple<const InDelta, const PsDelta...> deltas;
  };


  /**
   * Deduction guide
   */

  template<typename Trans, typename InDelta, typename ... PsDelta>
  FiniteDifferenceLinearization(Trans&&, InDelta&&, PsDelta&&...)
    -> FiniteDifferenceLinearization<Trans, InDelta, PsDelta...>;

}


#endif //OPENKALMAN_FINITEDIFFERENCELINEARIZATION_HPP
