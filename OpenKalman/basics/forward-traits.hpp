/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file Traits.h
 * A header file containing forward declarations for all OpenKalman traits.
 */

#ifndef OPENKALMAN_FORWARD_TRAITS_HPP
#define OPENKALMAN_FORWARD_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // ---------------- //
  //   Coefficients   //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * A type trait testing whether T is an atomic group of coefficients.
     *
     * The atomic coefficient groups are the following:
     * - Axis
     * - Circle (or alias Angle)
     * - Distance
     * - Inclination
     * - Polar
     * - Spherical
     * Atomic coefficient groups may be combined into composite coefficient sets by passing them as template
     * arguments to Coefficients. For example Coefficients<Axis, Polar<Distance, Angle>> is a set comprising an axis and
     * a set of polar coordinates.
     */
    template<typename T>
    struct is_atomic_coefficient_group;

    /**
     * \internal
     * A type trait testing whether T is a composite set of coefficient groups.
     *
     * This corresponds to any specialization of the class Coefficients. Composite coefficients can, themselves,
     * comprise groups of other composite components. For example, Coefficients<Axis, Coefficients<Axis, Angle>>
     * tests positive for is_composite_coefficients.
     */
    template<typename T>
    struct is_composite_coefficients;
  } // namespace internal


#ifndef __cpp_concepts
  namespace detail
  {
    // A type trait testing whether T is either an atomic group of coefficients, or a composite set of coefficients.
    template<typename T>
    struct is_coefficients : std::integral_constant<bool,
      internal::is_composite_coefficients<T>::value or internal::is_atomic_coefficient_group<T>::value> {};
  }
#endif


  /**
   * T is a coefficient group.
   *
   * A coefficient group may consist of some combination of any of the following:
   * - Axis
   * - Circle (including alias Angle)
   * - Distance
   * - Inclination
   * - Polar
   * - Spherical
   * - Coefficient (a composite coefficient including any other coefficient group).
   * Examples: Axis, Angle, Coefficient<Axis, Axis>, Coefficient<Angle, Coefficient<Axis, Angle>>.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept coefficients = internal::is_composite_coefficients<T>::value or
    internal::is_atomic_coefficient_group<T>::value;
#else
  template<typename T>
  inline constexpr bool coefficients = detail::is_coefficients<T>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * Type trait testing whether coefficients T are equivalent to coefficients U.
     */
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable = void>
#endif
    struct is_equivalent_to;
  }


  /**
   * T is equivalent to U.
   *
   * For example, <code>equivalent_to<Axis, Coefficients<Axis>></code> returns <code>true</code>.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept equivalent_to = internal::is_equivalent_to<T, U>::value;
#else
  inline constexpr bool equivalent_to = internal::is_equivalent_to<T, U>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * Type trait testing whether T (a set of coefficients) is a prefix of U.
     */
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable = void>
#endif
    struct is_prefix_of;
  }


  /**
   * T is a prefix of U
   *
   * For example, <code>prefix_of<Coefficients<Axis>, Coefficients<Axis, Angle>></code> returns <code>true</code>.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of = internal::is_prefix_of<T, U>::value;
#else
  inline constexpr bool prefix_of = internal::is_prefix_of<T, U>::value;
#endif


  // ------------------ //
  //   General traits   //
  // ------------------ //

  /**
   * Describes the traits of a matrix T, such as its dimensions, coefficient types, etc.
   * \addtogroup Traits
   * \tparam T The matrix type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename T>
  struct MatrixTraits {};

  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct MatrixTraits<T> : public MatrixTraits<std::decay_t<T>> {};
#else
  template<typename T, typename Enable = void> ///< \tparam Enable A dummy variable to enable the class.
  struct MatrixTraits {};

  template<typename T>
  struct MatrixTraits<T&> : MatrixTraits<T> {};

  template<typename T>
  struct MatrixTraits<T&&> : MatrixTraits<T> {};

  template<typename T>
  struct MatrixTraits<const T> : MatrixTraits<T> {};
#endif


  /**
   * Describes the traits of a distribution type.
   * \addtogroup Traits
   * \tparam T The distribution type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename T>
  struct DistributionTraits {};

  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct DistributionTraits<T> : DistributionTraits<std::decay_t<T>> {};
#else
  template<typename T, typename Enable = void> ///< \tparam Enable A dummy variable to enable the class.
  struct DistributionTraits {};

  template<typename T>
  struct DistributionTraits<T&> : DistributionTraits<T> {};

  template<typename T>
  struct DistributionTraits<T&&> : DistributionTraits<T> {};

  template<typename T>
  struct DistributionTraits<const T> : DistributionTraits<T> {};
#endif


  // --------------------- //
  //    covariance_base    //
  // --------------------- //

  namespace internal
  {
    /**
     * \internal
     * All true instances of is_covariance_base need to defined in each matrix interface.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_covariance_base : std::false_type {};
  }

  /**
   * T is an acceptable base matrix for a covariance (including square_root_covariance).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance_base = internal::is_covariance_base<std::decay_t<T>>::value;
#else
  inline constexpr bool covariance_base = internal::is_covariance_base<std::decay_t<T>>::value;
#endif


  // ----------------------- //
  //    typed_matrix_base    //
  // ----------------------- //

  namespace internal
  {
    /**
     * \internal
     * All true instances of is_typed_matrix_base need to be defined in each matrix interface.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_typed_matrix_base : std::false_type {};
  }

  /**
   * T is an acceptable base for a general typed matrix (e.g., matrix, mean, or euclidean_mean)
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix_base = internal::is_typed_matrix_base<std::decay_t<T>>::value;
#else
  inline constexpr bool typed_matrix_base = internal::is_typed_matrix_base<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_TRAITS_HPP
