/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_IDENTITYTRANSFORMATION_HPP
#define OPENKALMAN_IDENTITYTRANSFORMATION_HPP


namespace OpenKalman
{

  /**
   * @brief The identity transformation from one single-column vector to another.
   * Perturbation terms are treated as additive.
   */
  struct IdentityTransformation
  {
    /// Applies the transformation.
    template<typename In, typename ... Perturbations>
    auto operator()(In&& in, Perturbations&& ... ps) const noexcept
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(MatrixTraits<In>::columns == 1);
      static_assert(((internal::PerturbationTraits<Perturbations>::columns == 1) and ...));
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients,
        typename MatrixTraits<In>::RowCoefficients>...>);

      return strict((std::forward<In>(in) + ... + std::forward<Perturbations>(ps)));
    }

    /// Returns a tuple of the Jacobians for the input and each perturbation term.
    template<typename In, typename ... Perturbations>
    auto jacobian(const In&, const Perturbations&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(MatrixTraits<In>::columns == 1);
      static_assert(((internal::PerturbationTraits<Perturbations>::columns == 1) and ...));
      using Coeffs = typename MatrixTraits<In>::RowCoefficients;
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, Coeffs>...>);

      auto jacobian0 = MatrixTraits<In>::identity();
      auto jacobians = MatrixTraits<decltype(jacobian0)>::zero();
      return std::tuple_cat(std::tuple {jacobian0}, internal::tuple_replicate<sizeof...(Perturbations)>(jacobians));
    }

    /// Returns a tuple of Hessian matrices for the input and each perturbation term. In this case, they are zero matrices.
    template<typename In, typename ... Perturbations>
    auto hessian(const In&, const Perturbations&...) const
    {
      static_assert(is_column_vector_v<In>);
      static_assert((is_perturbation_v<Perturbations> and ...));
      static_assert(MatrixTraits<In>::columns == 1);
      static_assert(((internal::PerturbationTraits<Perturbations>::columns == 1) and ...));
      using Coeffs = typename MatrixTraits<In>::RowCoefficients;
      static_assert(std::conjunction_v<
        is_equivalent<typename internal::PerturbationTraits<Perturbations>::RowCoefficients, Coeffs>...>);

      return zero_hessian<Coeffs, In, Perturbations...>();
    }

  };

}


#endif //OPENKALMAN_IDENTITYTRANSFORMATION_HPP
