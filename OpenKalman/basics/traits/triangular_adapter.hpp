/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref triangular_adapter.
 */

#ifndef OPENKALMAN_TRIANGULAR_ADAPTER_HPP
#define OPENKALMAN_TRIANGULAR_ADAPTER_HPP


namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_triangular_adapter : std::false_type {};

    template<typename T>
    struct is_triangular_adapter<T, std::enable_if_t<interface::indexible_object_traits<std::decay_t<T>>::is_triangular_adapter>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a triangular adapter of triangle type triangle_type.
   * \details If T has a dynamic shape, it is not guaranteed to be triangular because it could be non-square.
   * \details A triangular adapter is necessarily triangular if it is a square matrix. If it is not a square matrix,
   * only the truncated square portion of the matrix would be triangular.
   * \tparam T A matrix or tensor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept triangular_adapter = interface::indexible_object_traits<std::decay_t<T>>::is_triangular_adapter and
#else
  constexpr bool triangular_adapter = detail::is_triangular_adapter<T>::value and has_nested_object<T> and
#endif
    has_nested_object<T> and square_shaped<T, Qualification::depends_on_dynamic_shape>;


} // namespace OpenKalman

#endif //OPENKALMAN_TRIANGULAR_ADAPTER_HPP
