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
 * \internal
 * \file
 * \brief Declaration for \ref is_modifiable.
 */

#ifndef OPENKALMAN_IS_MODIFIABLE_HPP
#define OPENKALMAN_IS_MODIFIABLE_HPP

#include <type_traits>


namespace OpenKalman::internal
{
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct has_const : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires std::is_const_v<std::remove_reference_t<T>> or
      (requires { typename nested_object_of_t<T>; } and has_const<nested_object_of_t<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_object_of_t<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      has_const<T>::value or
      (not maybe_same_shape_as<T, U>) or
      (not std::same_as<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (triangular_matrix<T, TriangleType::upper> and not triangular_matrix<U, TriangleType::upper>) or
      (triangular_matrix<T, TriangleType::lower> and not triangular_matrix<U, TriangleType::lower>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)
    struct is_modifiable<T, U> : std::false_type {};
#else
    template<typename T, typename U>
    struct is_modifiable<T, U, std::enable_if_t<
      has_const<T>::value or
      (not maybe_same_shape_as<T, U>) or
      (not std::is_same_v<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (triangular_matrix<T, TriangleType::upper> and not triangular_matrix<U, TriangleType::upper>) or
      (triangular_matrix<T, TriangleType::lower> and not triangular_matrix<U, TriangleType::lower>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)>> : std::false_type {};
#endif

} // namespace OpenKalman::internal

#endif //OPENKALMAN_IS_MODIFIABLE_HPP
