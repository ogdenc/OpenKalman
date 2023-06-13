/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general conversion functions.
 */

#ifndef OPENKALMAN_CONVERSION_FUNCTIONS_HPP
#define OPENKALMAN_CONVERSION_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // ================================== //
  //  Modular transformation functions  //
  // ================================== //

  /**
   * \brief Transform a matrix or tensor into Euclidean space along its first index.
   * \tparam Arg A matrix or tensor. I
   */
#ifdef __cpp_concepts
  template<wrappable Arg, index_descriptor C>
  requires dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
    equivalent_to<C, coefficient_types_of_t<Arg, 0>>
#else
  template<typename Arg, typename C, std::enable_if_t<wrappable<Arg> and index_descriptor<C> and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_index_descriptor<coefficient_types_of_t<Arg, 0>>)
      if (not get_index_descriptor_is_euclidean(get_dimensions_of<0>(arg)) and c != get_dimensions_of<0>(arg))
        throw std::domain_error {"In to_euclidean, specified index descriptor does not match that of the object's index 0"};

    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of to_euclidean is not wrappable"};

      return interface::ModularTransformationTraits<Arg>::to_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<all_fixed_indices_are_euclidean Arg>
#else
  template<typename Arg, std::enable_if_t<all_fixed_indices_are_euclidean<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    return to_euclidean(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


#ifdef __cpp_concepts
  template<wrappable Arg, index_descriptor C>
  requires (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
    equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<wrappable<Arg> and index_descriptor<C> and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_index_descriptor<coefficient_types_of_t<Arg, 0>>)
      if (not get_index_descriptor_is_euclidean(get_dimensions_of<0>(arg)) and c != get_dimensions_of<0>(arg))
        throw std::domain_error {"In from_euclidean, specified index descriptor does not match that of the object's index 0"};

    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of from_euclidean is not wrappable"};

      return interface::ModularTransformationTraits<Arg>::from_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<all_fixed_indices_are_euclidean Arg>
#else
  template<typename Arg, std::enable_if_t<all_fixed_indices_are_euclidean<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg)
  {
    return from_euclidean(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


#ifdef __cpp_concepts
  template<wrappable Arg, index_descriptor C>
  requires (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
    equivalent_to<C, coefficient_types_of_t<Arg, 0>>)
#else
  template<typename Arg, typename C, std::enable_if_t<wrappable<Arg> and index_descriptor<C> and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, coefficient_types_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg, const C& c)
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_index_descriptor<coefficient_types_of_t<Arg, 0>>)
      if (not get_index_descriptor_is_euclidean(get_dimensions_of<0>(arg)) and c != get_dimensions_of<0>(arg))
        throw std::domain_error {"In wrap_angles, specified index descriptor does not match that of the object's index 0"};

    if constexpr (euclidean_index_descriptor<C> or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of wrap_angles is not wrappable"};

      interface::ModularTransformationTraits<Arg>::wrap_angles(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<all_fixed_indices_are_euclidean Arg>
#else
  template<typename Arg, std::enable_if_t<all_fixed_indices_are_euclidean<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg)
  {
    return wrap_angles(std::forward<Arg>(arg), get_dimensions_of<0>(arg));
  }


  // ==================== //
  //  Internal functions  //
  // ==================== //

  namespace internal
  {
    // ------------------------ //
    //  to_covariance_nestable  //
    // ------------------------ //

    /**
     * \overload
     * \internal
     * \brief Convert a \ref covariance_nestable matrix or \ref typed_matrix_nestable to a \ref covariance_nestable.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))) and
      (row_dimension_of_v<Arg> == row_dimension_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>)
#else
    template<typename T, typename Arg, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))) and
      (row_dimension_of<Arg>::value == row_dimension_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \internal
     * \brief Convert \ref covariance or \ref typed_matrix to a \ref covariance_nestable of type T.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \tparam Arg A \ref covariance or \ref typed_matrix.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))) and
      (row_dimension_of_v<Arg> == row_dimension_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>)
#else
    template<typename T, typename Arg, typename = void, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and (not std::is_void_v<Arg>) and
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))) and
      (row_dimension_of<Arg>::value == row_dimension_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return The result of converting Arg to a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))
#else
    template<typename Arg, typename = std::enable_if_t<covariance_nestable<Arg> or
        (typed_matrix_nestable<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return A \ref triangular_matrix if Arg is a \ref triangular_covariance or otherwise a \ref hermitian_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))
#else
    template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or dimension_size_of_index_is<Arg, 1, 1>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_CONVERSION_FUNCTIONS_HPP
