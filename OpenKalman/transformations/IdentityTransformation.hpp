/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for IdentityTransformation.
 */

#ifndef OPENKALMAN_IDENTITYTRANSFORMATION_HPP
#define OPENKALMAN_IDENTITYTRANSFORMATION_HPP


namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


  /**
   * \brief The identity tests from one single-column vector to another.
   * \details Perturbation terms are treated as additive.
   */
  struct IdentityTransformation;


  namespace internal
  {
#ifdef __cpp_concepts
    template<std::size_t order> requires (order <= 1)
    struct is_linearized_function<IdentityTransformation, order> : std::true_type {};
#else
  template<std::size_t order>
    struct is_linearized_function<IdentityTransformation, order, std::enable_if_t<order <= 1>> : std::true_type {};
#endif
  }


  struct IdentityTransformation
  {
    /// Applies the tests.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation<vector_space_descriptor_of_t<In, 0>> ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations, vector_space_descriptor_of_t<In, 0>> and ...), int> = 0>
#endif
    auto operator()(In&& in, Perturbations&& ... ps) const noexcept
    {
      return make_self_contained((std::forward<In>(in) + ... + std::forward<Perturbations>(ps)));
    }


    /// Returns a tuple of the Jacobians for the input and each perturbation term.
#ifdef __cpp_concepts
    template<transformation_input In, perturbation<vector_space_descriptor_of_t<In, 0>> ... Perturbations>
#else
    template<typename In, typename ... Perturbations, std::enable_if_t<transformation_input<In> and
      (perturbation<Perturbations, vector_space_descriptor_of_t<In, 0>> and ...), int> = 0>
#endif
    auto jacobian(In&& in, Perturbations&&...ps) const
    {
      return std::make_tuple(
        make_identity_matrix_like(in),
        make_zero_matrix_like(make_identity_matrix_like(ps))...);
    }

  };

}


#endif //OPENKALMAN_IDENTITYTRANSFORMATION_HPP
