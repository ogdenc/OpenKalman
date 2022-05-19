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

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_INTERFACE_TRAITS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_INTERFACE_TRAITS_HPP

#include <type_traits>

namespace OpenKalman::interface
{
  // ----------------------- //
  //   IndexDescriptorSize   //
  // ----------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor.
   * \details Instances may inherit from <code>std::integral_constant<std::size_t, ...>. They must define
   * the following as static members:
   * \code
   * /// The number of dimensions (or \ref dynamic_size if not known at compile time).
   * static constexpr std::size_t value = a_compile_time_size;
   *
   * /// Get the dimension size, at runtime, of an \ref index_descriptor.
   * static constexpr std::size_t get(const std::decay_t<T>& t) { return a_runtime_size; };
   * \endcode
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct IndexDescriptorSize;


  /// \overload An integral value is an dynamically-sized index descriptor.
#ifdef __cpp_concepts
  template<std::integral T>
  struct IndexDescriptorSize<T>
#else
  template<typename T>
  struct IndexDescriptorSize<T, std::enable_if_t<std::is_integral_v<T>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size>
  {
    static constexpr std::size_t get(const std::decay_t<T>& t) { return t; }
  };


  // ------------------------------- //
  //   EuclideanIndexDescriptorSize   //
  // ------------------------------- //

  /**
   * \brief The dimension size of an \ref index_descriptor if it is transformed into Euclidean space.
   * \details Instances may inherit from <code>std::integral_constant<std::size_t, ...>. They must define
   * the following as static members:
   * \code
   * /// The number of dimensions when transformed to Euclidean space (or \ref dynamic_size if not known at compile time).
   * static constexpr std::size_t value = a_compile_time_size;
   *
   * /// Get the dimension size, at runtime (if transformed to Euclidean space), of an \ref index_descriptor.
   * static constexpr std::size_t get(const std::decay_t<T>& t) { return a_runtime_size; };
   * \endcode
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename Enable = void>
#endif
  struct EuclideanIndexDescriptorSize;


  /// \overload An integral value is an dynamically-sized index descriptor.
#ifdef __cpp_concepts
  template<std::integral T>
  struct EuclideanIndexDescriptorSize<T>
#else
  template<typename T>
  struct EuclideanIndexDescriptorSize<T, std::enable_if_t<std::is_integral_v<T>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size>
  {
    static constexpr std::size_t get(const std::decay_t<T>& t) { return t; }
  };


  // --------------------------------- //
  //   IndexDescriptorComponentCount   //
  // --------------------------------- //

  /**
   * \brief The number of atomic component parts of an \ref index_descriptor.
   * \details Instances may inherit from <code>std::integral_constant<std::size_t, ...>. They must define
   * the following as static members:
   * \code
   * /// The number of atomic component parts (or \ref dynamic_size if not known at compile time).
   * static constexpr std::size_t value = a_compile_time_size;
   *
   * /// Get the number of atomic component parts, at runtime, of an \ref index_descriptor.
   * static constexpr std::size_t get(const std::decay_t<T>& t) { return a_runtime_size; };
   * \endcode
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct IndexDescriptorComponentCount;


  /// \overload An integral value is an dynamically-sized index descriptor.
#ifdef __cpp_concepts
  template<std::integral T>
  struct IndexDescriptorComponentCount<T>
#else
  template<typename T>
  struct IndexDescriptorComponentCount<T, std::enable_if_t<std::is_integral_v<T>>>
#endif
    : std::integral_constant<std::size_t, dynamic_size>
  {
    static constexpr std::size_t get(const std::decay_t<T>& t) { return t; }
  };


  // --------------------------------- //
  //   IndexDescriptorDifferenceType   //
  // --------------------------------- //

  /**
   * \brief The type of the \ref index_descriptor when tensors having respective index_descriptors T are subtracted.
   * \details For example, subtracting two 1D vectors of type Direction yields a 1D vector of type Axis, so if
   * <code>T</code> is Distance, the resulting <code>type</code> will be Axis. Instances must define
   * the following as a static member:
   * \code
   * /// The difference type.
   * using type = a_difference_type;
   * \endcode
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct IndexDescriptorDifferenceType;


  /// \overload An integral value is an dynamically-sized index descriptor.
#ifdef __cpp_concepts
  template<std::integral T>
  struct IndexDescriptorDifferenceType<T>
#else
  template<typename T>
  struct IndexDescriptorDifferenceType<T, std::enable_if_t<std::is_integral_v<T>>>
#endif
  {
    using type = std::decay_t<T>;
  };


  // ---------------------------- //
  //   IndexDescriptorIsUntyped   //
  // ---------------------------- //

  /**
   * \brief Indicates whether \ref index_descriptor T is untyped.
   * \details By default, index descriptors are not untyped. An untyped index descriptor must may inherit from
   * <code>std::bool_constant<...>, and must define the following as static members:
   * \code
   * /// Whether T is known to be untyped at compile time.
   * static constexpr bool value = a_compile_time_bool_value;
   *
   * /// Whether index descriptor t is untyped at runtime.
   * static constexpr bool get(const std::decay_t<T>& t) { return a_runtime_bool_value; };
   * \endcode
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct IndexDescriptorIsUntyped
    : std::bool_constant<false>
  {
    static constexpr bool get(const std::decay_t<T>& t) { return false; }
  };


  /// \overload An integral value is an untyped index descriptor.
#ifdef __cpp_concepts
  template<std::integral T>
  struct IndexDescriptorIsUntyped<T>
#else
  template<typename T>
  struct IndexDescriptorIsUntyped<T, std::enable_if_t<std::is_integral_v<T>>>
#endif
    : std::bool_constant<true>
  {
    static constexpr bool get(const std::decay_t<T>& t) { return true; }
  };


} // namespace OpenKalman::interface

#endif //OPENKALMAN_INDEX_DESCRIPTOR_INTERFACE_TRAITS_HPP
