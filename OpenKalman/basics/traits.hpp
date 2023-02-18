/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Declarations for OpenKalman and native-matrix traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  using namespace interface;

  // --------------- //
  //  typed_adapter  //
  // --------------- //

  /**
   * \brief Specifies that T is a typed adapter expression.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_adapter =
#else
  constexpr bool typed_adapter =
#endif
    typed_matrix<T> or covariance<T> or euclidean_expr<T>;


  // ----------------- //
  //  untyped_adapter  //
  // ----------------- //

  /**
   * \brief Specifies that T is an untyped adapter expression.
   * \details Untyped adapter expressions are generally used whenever the native matrix library does not have an
   * important built-in matrix type, such as a diagonal matrix, a triangular matrix, or a hermitian matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept untyped_adapter =
#else
  constexpr bool untyped_adapter =
#endif
    eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T>;


  // ========= //
  //  Aliases  //
  // ========= //

  // ------------------------- //
  //  dense_writable_matrix_t  //
  // ------------------------- //

  namespace detail
  {
    template<typename T, typename Scalar, typename...D>
    struct dense_writable_matrix_impl
    {
      using type = std::decay_t<decltype(make_default_dense_writable_matrix_like<T, Scalar>(std::declval<D>()...))>;
    };


    template<typename T, typename Scalar>
    struct dense_writable_matrix_impl<T, Scalar>
    {
      using type = std::decay_t<decltype(make_default_dense_writable_matrix_like<Scalar>(std::declval<T>()))>;
    };
  }


  /**
    * \brief An alias for a dense, writable matrix, patterned on parameter T.
    * \tparam T A matrix or array from the relevant matrix library.
    * \tparam S A scalar type (may or may not be </code>scalar_type_of_t<T></code>.
    * \tparam D Index descriptors defining the dimensions of the new matrix.
    * \todo Create typed Matrix if Ds are typed.
    */
#ifdef __cpp_concepts
  template<indexible T, scalar_type S = scalar_type_of_t<T>, index_descriptor...D>
#else
  template<typename T, typename S = scalar_type_of_t<T>, typename...D>
#endif
  using dense_writable_matrix_t = typename detail::dense_writable_matrix_impl<T, std::decay_t<S>, D...>::type;


  // --------------------------------- //
  //  untyped_dense_writable_matrix_t  //
  // --------------------------------- //

  /**
   * \brief An alias for a dense, writable matrix, patterned on parameter T.
   * \tparam T A matrix or array from the relevant matrix library.
   * \tparam D Integral values defining the dimensions of the new matrix.
   */
#ifdef __cpp_concepts
  template<indexible T, scalar_type S, auto...D> requires ((std::is_integral_v<decltype(D)> and D >= 0) and ...)
#else
  template<typename T, typename S, auto...D>
#endif
  using untyped_dense_writable_matrix_t = dense_writable_matrix_t<T, S, Dimensions<static_cast<const std::size_t>(D)>...>;


  // --------------------------- //
  //  equivalent_self_contained  //
  // --------------------------- //

  /**
   * \brief An alias for type, derived from and equivalent to parameter T, that is self-contained.
   * \details Use this alias to obtain a type, equivalent to T, that can safely be returned from a function.
   * \sa self_contained, make_self_contained
   * \internal \sa interface::Dependencies
   */
  template<typename T>
  using equivalent_self_contained_t = std::remove_reference_t<decltype(make_self_contained(std::declval<T>()))>;


  // ------------ //
  //  passable_t  //
  // ------------ //

  /**
   * \brief An alias for a type, derived from and equivalent to parameter T, that can be passed as a function parameter.
   * \tparam T The type in question.
   * \details A passable type T is either an lvalue reference or is \ref equivalent_self_contained_t.
   */
  template<typename T>
  using passable_t = std::conditional_t<std::is_lvalue_reference_v<T>, T, equivalent_self_contained_t<T>>;


  // ========================================================================== //
  //  Traits for which specializations must be defined in the matrix interface  //
  // ========================================================================== //

  // --------------- //
  //  is_modifiable  //
  // --------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct has_const : std::false_type {};


#ifdef __cpp_concepts
    template<typename T> requires std::is_const_v<std::remove_reference_t<T>> or
      (requires { typename nested_matrix_of_t<T>; } and has_const<nested_matrix_of_t<T>>::value)
    struct has_const<T> : std::true_type {};
#else
    template<typename T>
    struct has_const<T, std::enable_if_t<std::is_const_v<std::remove_reference_t<T>>>> : std::true_type {};

    template<typename T>
    struct has_const<T, std::enable_if_t<(not std::is_const_v<std::remove_reference_t<T>>) and
      has_const<nested_matrix_of_t<T>>::value>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T, typename U> requires
      has_const<T>::value or
      (not maybe_has_same_shape_as<T, U>) or
      (not std::same_as<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (upper_triangular_matrix<T> and not upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and not lower_triangular_matrix<U>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)
    struct is_modifiable<T, U> : std::false_type {};
#else
    template<typename T, typename U>
    struct is_modifiable<T, U, std::enable_if_t<
      has_const<T>::value or
      (not maybe_has_same_shape_as<T, U>) or
      (not std::is_same_v<scalar_type_of_t<T>, scalar_type_of_t<U>>) or
      (constant_matrix<T> and not constant_matrix<U>) or
      (identity_matrix<T> and not identity_matrix<U>) or
      (upper_triangular_matrix<T> and not upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and not lower_triangular_matrix<U>) or
      (hermitian_matrix<T> and not hermitian_matrix<U>)>> : std::false_type {};
#endif

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_TRAITS_HPP
