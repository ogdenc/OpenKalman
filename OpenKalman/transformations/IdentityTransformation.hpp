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
#ifdef __cpp_concepts
    template<column_vector In, perturbation ... Perturbations> requires
      internal::transformation_args<In, Perturbations...>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      column_vector<In> and (perturbation<Perturbations> and ...) and
      internal::transformation_args<In, Perturbations...>, int> = 0>
#endif
    auto operator()(In&& in, Perturbations&& ... ps) const noexcept
    {
      return make_self_contained((std::forward<In>(in) + ... + std::forward<Perturbations>(ps)));
    }


    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<column_vector In, perturbation ... Perturbations> requires
    internal::transformation_args<In, Perturbations...>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      column_vector<In> and (perturbation<Perturbations> and ...) and
      internal::transformation_args<In, Perturbations...>, int> = 0>
#endif
    auto jacobian(const In&, const Perturbations&...) const
    {
      auto jacobian0 = MatrixTraits<In>::identity();
      auto jacobians = MatrixTraits<decltype(jacobian0)>::zero();
      return std::tuple_cat(std::tuple {jacobian0}, internal::tuple_replicate<sizeof...(Perturbations)>(jacobians));
    }


    /// Returns a tuple of Hessian matrices for the input and each perturbation term. In this case, they are zero matrices.
#ifdef __cpp_concepts
    template<column_vector In, perturbation ... Perturbations> requires
    internal::transformation_args<In, Perturbations...>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<
      column_vector<In> and (perturbation<Perturbations> and ...) and
      internal::transformation_args<In, Perturbations...>, int> = 0>
#endif
    auto hessian(const In&, const Perturbations&...) const
    {
      using Coeffs = typename MatrixTraits<In>::RowCoefficients;
      return zero_hessian<Coeffs, In, Perturbations...>();
    }

  };

}


#endif //OPENKALMAN_IDENTITYTRANSFORMATION_HPP
