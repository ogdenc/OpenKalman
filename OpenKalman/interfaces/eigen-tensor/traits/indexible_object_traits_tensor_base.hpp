/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Base for type traits as applied to native Eigen tensor types.
 */

#ifndef OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_TENSOR_BASE_HPP
#define OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_TENSOR_BASE_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<Eigen3::eigen_tensor_general T>
  struct indexible_object_traits_tensor_base<T>
#else
  template<typename T>
  struct indexible_object_traits_tensor_base<T, std::enable_if_t<Eigen3::eigen_tensor_general<T>>>
#endif
  {
    using scalar_type = typename Eigen::internal::traits<T>::Scalar;


    template<typename Arg>
    static constexpr auto
    count_indices(const Arg& arg)
    {
      return std::integral_constant<std::size_t, Eigen::internal::traits<T>::NumDimensions>{};
    }

  };


} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_TENSOR_BASE_HPP
