/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the Axis class.
 */

#ifndef OPENKALMAN_AXIS_HPP
#define OPENKALMAN_AXIS_HPP

#include <cmath>
#include <array>
#include <functional>

namespace OpenKalman
{
  /**
   * \brief A real or integral number, (&minus;&infin;,&infin;).
   * \details This is the default coefficient type. No wrapping occurs, and matrices operate as usual.
   * \internal
   * <b>See also</b> the following functions for accessing coefficient properties:
   * - internal::to_euclidean_element: \copybrief internal::to_euclidean_element
   * - internal::from_euclidean_element: \copybrief internal::from_euclidean_element
   * - internal::wrap_get_element: \copybrief internal::wrap_get_element
   * - internal::wrap_set_element \copybrief internal::wrap_set_element
   */
  struct Axis
  {
    /**
     * \internal
     * \brief A function taking a row index and returning a corresponding matrix element.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    template<typename Scalar>
    using GetCoeff = std::function<Scalar(const std::size_t)>;


    /**
     * \internal
     * \brief A function that sets a matrix element corresponding to a row index to a scalar value.
     * \details A separate function will be constructed for each column in the matrix.
     * \tparam Scalar The scalar type of the matrix.
     */
    template<typename Scalar>
    using SetCoeff = std::function<void(const std::size_t, const Scalar)>;


    /**
     * \internal
     * \brief An array of functions (here, just one) that transform an axis coefficient to Euclidean space.
     * \details Because an axis already represents a point in Euclidean space, this is an identity function.
     * Each array element is a function taking a ''get coefficient'' function and returning a coordinate value.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should generally be accessed only through \ref to_euclidean_element.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam start The index of the coefficient that is being transformed.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<scalar_type Scalar, std::size_t start>
#else
    template<typename Scalar, std::size_t start>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), 1>
    to_euclidean_array =
    {
      [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(start); }
    };


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires std::floating_point<scalar_type_of_t<T>> and (sizeof...(Is) == max_indices_of_v<T> - 1)
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and
      std::is_floating_point<typename scalar_type_of<T>::type>::value and
      (sizeof...(Is) == max_indices_of<T>::value - 1), int> = 0>
#endif
    static constexpr scalar_type_of_t<T>
    to_euclidean_element(const T& t, std::size_t euclidean_local_index, std::size_t start, std::size_t i, Is...is)
    {
      return get_element(t, start, is...);
    };


    /**
     * \internal
     * \brief An array of functions (here, just one) that transform a coordinate in Euclidean space into an axis.
     * \details Because an axis already represents a point in Euclidean space, this is an identity function.
     * The array element is a function taking a ''get coefficient'' function and returning an axis.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the value.
     * \note This should generally be accessed only through \ref internal::from_euclidean_element.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam euclidean_start The index of the coefficient that is being transformed.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<scalar_type Scalar, std::size_t euclidean_start>
#else
    template<typename Scalar, std::size_t euclidean_start>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), 1>
      from_euclidean_array =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(euclidean_start); }
      };


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires std::floating_point<scalar_type_of_t<T>> and (sizeof...(Is) == max_indices_of_v<T> - 1)
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and
      std::is_floating_point<typename scalar_type_of<T>::type>::value and
      (sizeof...(Is) == max_indices_of<T>::value - 1), int> = 0>
#endif
    static constexpr scalar_type_of_t<T>
    from_euclidean_element(const T& t, std::size_t local_index, std::size_t euclidean_start, std::size_t i, Is...is)
    {
      return get_element(t, euclidean_start, is...);
    };


    /**
     * \internal
     * \brief An array of functions (here, just one) that return a wrapped version of an axis.
     * \details Each function in the array takes a ''get coefficient'' function and returning an (unchanged) axis.
     * The ''get coefficient'' function takes the index of a column within a row vector and returns the coefficient.
     * \note This should generally be accessed only through \ref internal::wrap_get_element.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam start The index of the axis coefficient that is being wrapped.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<scalar_type Scalar, std::size_t start>
#else
    template<typename Scalar, std::size_t start>
#endif
    static constexpr std::array<Scalar (*const)(const GetCoeff<Scalar>&), 1>
      wrap_array_get =
      {
        [](const GetCoeff<Scalar>& get_coeff) { return get_coeff(start); }
      };


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires std::floating_point<scalar_type_of_t<T>> and (sizeof...(Is) == max_indices_of_v<T>)
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and
      std::is_floating_point<typename scalar_type_of<T>::type>::value and
      (sizeof...(Is) == max_indices_of<T>::value), int> = 0>
#endif
    static constexpr scalar_type_of_t<T>
    wrap_get_element(const T& t, std::size_t local_index, Is...is)
    {
      return get_element(t, is...);
    };


    /**
     * \internal
     * \brief An array of functions (here, just one) that sets a matrix coefficient to an axis value.
     * \details Each void function in the array takes a scalar value and ''set coefficient'' function.
     * The ''set coefficient'' function takes a scalar value and an index of a column within a row vector and
     * sets the coefficient at that index to that (unchanged) scalar input.
     * \note This should generally be accessed only through \ref internal::wrap_set_element.
     * \tparam Scalar The scalar type (e.g., double).
     * \tparam start The index of the axis coefficient that is being wrapped.
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<scalar_type Scalar, std::size_t start>
#else
    template<typename Scalar, std::size_t start>
#endif
    static constexpr
      std::array<void (*const)(const Scalar, const SetCoeff<Scalar>&, const GetCoeff<Scalar>&), 1>
      wrap_array_set =
      {
        [](const Scalar s, const SetCoeff<Scalar>& set_coeff, const GetCoeff<Scalar>&) { set_coeff(start, s); }
      };


#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<indexible T, std::convertible_to<const std::size_t>...Is>
    requires std::floating_point<scalar_type_of_t<T>> and (sizeof...(Is) == max_indices_of_v<T>)
#else
    template<typename T, typename...Is, std::enable_if_t<indexible<T> and
      (std::is_convertible<Is, const std::size_t>::value and ...) and
      std::is_floating_point<typename scalar_type_of<T>::type>::value and
      (sizeof...(Is) == max_indices_of<T>::value), int> = 0>
#endif
    static constexpr void
    wrap_set_element(T& t, scalar_type_of_t<T> s, std::size_t local_index, Is...is)
    {
      set_element(t, s, is...);
    };

  };


  /**
   * \brief Axis is represented by one coordinate.
   */
  template<>
  struct dimension_size_of<Axis> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Axis&) { return 1; }
  };


  /**
   * \brief Axis is represented by one coordinate in Euclidean space.
   */
  template<>
  struct euclidean_dimension_size_of<Axis> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Axis&) { return 1; }
  };


  /**
   * \brief The number of atomic components.
   */
  template<>
  struct index_descriptor_components_of<Axis> : std::integral_constant<std::size_t, 1>
  {
    constexpr static std::size_t get(const Axis&) { return 1; }
  };


  /**
   * \brief The type of the result when subtracting two Axis values.
   * \details A difference between two Axis values is also on an Axis, so there is no wrapping.
   */
  template<>
  struct dimension_difference_of<Axis> { using type = Axis; };


  template<>
  struct is_untyped_index_descriptor<Axis> : std::true_type {};


} // namespace OpenKalman


#endif //OPENKALMAN_AXIS_HPP
