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
 * \brief Traits for index descriptors.
 */

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // --------------------- //
  //   dimension_size_of   //
  // --------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor.
   * \details Instances should inherit from <code>std::integral_constant<std::size_t, ...> or at least define
   * static constexpr member <code>value</code>. Instances must also define static member function
   * <code>get(const T& t)</code> reflecting the size of t at runtime.
   * \note There is no need to define instances where T is cv- or ref-qualified.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_size_of
  {
    /**
     * \brief The number of dimensions (or \ref dynamic_size if not known at compile time).
     */
    std::size_t value = 0;

    /**
     * /brief Get the dimension size, at runtime, of an \ref index_descriptor.
     * \param t An \ref index_descriptor
     * \note This must be defined for T to be recognized as an \ref index_descriptor.
     */
    static constexpr std::size_t get(const std::decay_t<T>& t) = delete;
  };


#ifdef __cpp_concepts
  template<typename T> requires std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<T>
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<T>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size>
  {
    static constexpr std::size_t get(const std::decay_t<T>& t) { return t; }
  };


  // Note: dimension_size_of for non-integral index_descriptor types are defined elsewhere.


#ifdef __cpp_concepts
  template<typename T> requires (not std::is_same_v<T, std::decay_t<T>>)
  struct dimension_size_of<T>
#else
  template<typename T>
  struct dimension_size_of<T, std::enable_if_t<not std::is_same_v<T, std::decay_t<T>>>>
#endif
    : dimension_size_of<std::decay_t<T>> {};


  /**
   * \brief Helper template for \ref dimension_size_of.
   */
  template<typename T>
  constexpr auto dimension_size_of_v = dimension_size_of<std::decay_t<T>>::value;


  // ------------------------------- //
  //   euclidean_dimension_size_of   //
  // ------------------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor if it is transformed into Euclidean space.
   * \details Instances should inherit from <code>std::integral_constant<std::size_t, ...> or at least define
   * static constexpr member <code>value</code>. Instances must also define static member function
   * <code>get(const T& t)</code> reflecting the size (if transformed to Euclidean space) of t at runtime.
   * \note There is no need to define instances where T is cv- or ref-qualified.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename Enable = void>
#endif
  struct euclidean_dimension_size_of
  {
    /**
     * /brief Get the dimension size, at runtime (if transformed to Euclidean space), of an \ref index_descriptor.
     * \param t An \ref index_descriptor
     * \note This must be defined for T to be recognized as an \ref index_descriptor.
     */
    static constexpr std::size_t get(const std::decay_t<T>& t) = delete;
  };


#ifdef __cpp_concepts
  template<typename T> requires std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<std::decay_t<T>>
  struct euclidean_dimension_size_of<T>
#else
  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<
    std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<std::decay_t<T>>>>
#endif
    : dimension_size_of<T> {};


  // Note: euclidean_dimension_size_of for non-integral index_descriptor types are defined elsewhere.


#ifdef __cpp_concepts
  template<typename T> requires (not std::is_same_v<T, std::decay_t<T>>)
  struct euclidean_dimension_size_of<T>
#else
  template<typename T>
  struct euclidean_dimension_size_of<T, std::enable_if_t<not std::is_same_v<T, std::decay_t<T>>>>
#endif
    : euclidean_dimension_size_of<std::decay_t<T>> {};


  /**
   * \brief Helper template for \ref euclidean_dimension_size_of.
   */
  template<typename T>
  constexpr auto euclidean_dimension_size_of_v = euclidean_dimension_size_of<std::decay_t<T>>::value;


  // ---------------------------------- //
  //   index_descriptor_components_of   //
  // ---------------------------------- //

  /**
   * \brief The number of atomic component parts of an \ref index_descriptor.
   * \details Instances should inherit from <code>std::integral_constant<std::size_t, ...> or at least define
   * static constexpr member <code>value</code>. Instances must also define static member function
   * <code>get(const T& t)</code> reflecting the number of components of t at runtime.
   * \note There is no need to define instances where T is cv- or ref-qualified.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct index_descriptor_components_of
  {
    /**
     * /brief Get the number of atomic component parts, at runtime, of an \ref index_descriptor.
     * \param t An \ref index_descriptor
     * \note This must be defined for T to be recognized as an \ref index_descriptor.
     */
    static constexpr std::size_t get(const std::decay_t<T>& t) = delete;
  };


#ifdef __cpp_concepts
  template<typename T> requires std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<T>
  struct index_descriptor_components_of<T>
#else
  template<typename T>
  struct index_descriptor_components_of<T, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<T>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size>
  {
    static constexpr std::size_t get(const std::decay_t<T>& t) { return t; }
  };


  // Note: dimension_size_of for non-integral index_descriptor types are defined elsewhere.


#ifdef __cpp_concepts
  template<typename T> requires (not std::is_same_v<T, std::decay_t<T>>)
  struct index_descriptor_components_of<T>
#else
  template<typename T>
  struct index_descriptor_components_of<T, std::enable_if_t<not std::is_same_v<T, std::decay_t<T>>>>
#endif
    : index_descriptor_components_of<std::decay_t<T>> {};


  /**
   * \brief Helper template for \ref index_descriptor_components_of.
   */
  template<typename T>
  constexpr auto index_descriptor_components_of_v = index_descriptor_components_of<std::decay_t<T>>::value;


  // --------------------------- //
  //   dimension_difference_of   //
  // --------------------------- //

  /**
   * \brief The type of the \ref index_descriptor when tensors having respective index_descriptors T are subtracted.
   * \details For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis, so if
   * <code>T</code> is Distance, the resulting <code>type</code> will be Axis.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct dimension_difference_of {};


#ifdef __cpp_concepts
  template<typename T> requires std::is_integral_v<std::decay_t<T>>
  struct dimension_difference_of<T>
#else
  template<typename T>
  struct dimension_difference_of<T, std::enable_if_t<std::is_integral_v<std::decay_t<T>>>>
#endif
  {
    using type = std::decay_t<T>;
  };


  // Note: dimension_difference_of for non-integral index_descriptor types are defined elsewhere.


  /**
   * \brief Helper template for \ref dimension_difference_of.
   */
  template<typename T>
  using dimension_difference_of_t = typename dimension_difference_of<std::decay_t<T>>::type;


  // ------------------------------- //
  //   is_untyped_index_descriptor   //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct is_untyped_index_descriptor : std::false_type {};


#ifdef __cpp_concepts
  template<typename T> requires std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<T>
  struct is_untyped_index_descriptor<T>
#else
  template<typename T>
  struct is_untyped_index_descriptor<T, std::enable_if_t<std::is_same_v<T, std::decay_t<T>> and std::is_integral_v<T>>>
#endif
    : std::true_type {};


  // Note: is_untyped_index_descriptor for non-integral index_descriptor types are defined elsewhere.


#ifdef __cpp_concepts
  template<typename T> requires (not std::is_same_v<T, std::decay_t<T>>)
  struct is_untyped_index_descriptor<T>
#else
  template<typename T>
  struct is_untyped_index_descriptor<T, std::enable_if_t<not std::is_same_v<T, std::decay_t<T>>>>
#endif
    : is_untyped_index_descriptor<std::decay_t<T>> {};


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_DESCRIPTOR_TRAITS_HPP
