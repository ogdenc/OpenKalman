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
 * \brief Definition for \ref hermitian_adapter.
 */

#ifndef OPENKALMAN_HERMITIAN_ADAPTER_HPP
#define OPENKALMAN_HERMITIAN_ADAPTER_HPP

#include "linear-algebra/enumerations.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, HermitianAdapterType t, typename = void>
    struct hermitian_adapter_impl : std::false_type {};

    template<typename T, HermitianAdapterType t>
    struct hermitian_adapter_impl<T, t, std::enable_if_t<
      (t == HermitianAdapterType::any ?
       interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type == HermitianAdapterType::lower or
         interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type == HermitianAdapterType::upper :
       interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type == t)>> : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix adapter of a particular type.
   * \details A hermitian adapter may or may not actually be a \ref hermitian_matrix, depending on whether it is a \ref square_shaped.
   * If it is not a square matrix, it can still be a hermitian adapter, but only the truncated
   * square portion of the matrix would be hermitian.
   * \tparam T A matrix or tensor.
   * \tparam t The HermitianAdapterType of T.
   */
  template<typename T, HermitianAdapterType t = HermitianAdapterType::any>
#ifdef __cpp_concepts
  concept hermitian_adapter = hermitian_matrix<T> and
    (t == HermitianAdapterType::any ?
     interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type == HermitianAdapterType::lower or
       interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type == HermitianAdapterType::upper :
     interface::object_traits<stdex::remove_cvref_t<T>>::hermitian_adapter_type == t);
#else
  constexpr bool hermitian_adapter = hermitian_matrix<T> and detail::hermitian_adapter_impl<std::decay_t<T>, t>::value;
#endif


}

#endif
