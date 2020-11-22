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
 * \file forward-traits.hpp
 * A header file containing forward declarations for traits relating to OpenKalman or native matrix types.
 */

#ifndef OPENKALMAN_FORWARD_TRAITS_HPP
#define OPENKALMAN_FORWARD_TRAITS_HPP

#include <type_traits>


/**
 * The root namespace for OpenKalman.
 */
namespace OpenKalman
{
  // ---------------- //
  //   Coefficients   //
  // ---------------- //

  /**
   * \internal
   * Internal definitions, not intended for use outside of OpenKalman.
   */
  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether T is an atomic group of coefficients.
     * \details Atomic coefficient groups are %coefficients or groups of %coefficients that function as a unit,
     * and cannot be separated. They may be combined into composite coefficients by passing them as template
     * parameters to Coefficients.
     */
    template<typename T>
    struct is_atomic_coefficient_group;


    /**
     * \internal
     * \brief A type trait testing whether T is a composite set of coefficient groups.
     * \details Composite coefficients are specializations of the class Coefficients, which has the purpose of grouping
     * other atomic or composite coefficients. Composite coefficients can, themselves, comprise groups of other
     * composite components.
     */
    template<typename T>
    struct is_composite_coefficients;


    /**
     * \internal
     * A sufficiently-defined class representing an atomic or composite group of coefficients.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept coefficient_class =
    std::integral<decltype(T::size)> and std::integral<decltype(T::dimension)> and
      (T::axes_only or not T::axes_only) and
      (internal::is_atomic_coefficient_group<typename T::difference_type>::value or
        internal::is_composite_coefficients<typename T::difference_type>::value) and
      requires {T::template to_Euclidean_array<double, 0>[0](std::function<double(const std::size_t)>()) == 0.;} and
      requires {T::template from_Euclidean_array<double, 0>[0](std::function<double(const std::size_t)>()) == 0.;} and
      requires {T::template wrap_array_get<double, 0>[0](std::function<double(const std::size_t)>()) == 0.;} and
      requires {T::template wrap_array_set<double, 0>[0](0., std::function<void(const double, const std::size_t)>(),
        std::function<double(const std::size_t)>());} and
      (std::tuple_size_v<decltype(T::template to_Euclidean_array<double, 0>)> == T::dimension) and
      (std::tuple_size_v<decltype(T::template from_Euclidean_array<double, 0>)> == T::size) and
      (std::tuple_size_v<decltype(T::template wrap_array_get<double, 0>)> == T::size) and
      (std::tuple_size_v<decltype(T::template wrap_array_set<double, 0>)> == T::size);
#else
    inline constexpr bool coefficient_class = true;
#endif

  }


  /**
   * \interface coefficients<>
   * \brief T is a group of atomic or composite coefficients.
   * \copydetails internal::is_atomic_coefficient_group<>
   * \copydetails internal::is_composite_coefficients<>
   * \details Examples of coefficients:
   * - Axis
   * - Polar<Distance, angle::Radians>
   * - Coefficients<Axis, angle::Radians>
   * - Coefficients<Spherical<angle::Degrees, inclination::degrees, Distance>, Axis, Axis>
   */
#ifdef __cpp_concepts
  template<typename T>
  concept coefficients = internal::is_composite_coefficients<T>::value or
    internal::is_atomic_coefficient_group<T>::value;
#else
  namespace detail
  {
    // A type trait testing whether T is either an atomic group of coefficients, or a composite set of coefficients.
    template<typename T>
    struct is_coefficients : std::integral_constant<bool,
      internal::is_composite_coefficients<T>::value or internal::is_atomic_coefficient_group<T>::value> {};
  }

  template<typename T>
  inline constexpr bool coefficients = detail::is_coefficients<T>::value;
#endif


  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether coefficients T are equivalent to coefficients U.
     * \details Sets of coefficients are equivalent if they are treated functionally the same.
     */
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable = void>
    ///< \tparam Enable A dummy parameter for selection with SFINAE.
#endif
    struct is_equivalent_to;
  }


  /**
   * \interface equivalent_to<>
   * \brief T is equivalent to U, where T and U are sets of coefficients.
   * \copydetails internal::is_equivalent_to<>.
   * \details For example, <code>equivalent_to<Axis, Coefficients<Axis>></code> returns <code>true</code>.
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
     * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
     */
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable = void>
    ///< \tparam Enable A dummy parameter for selection with SFINAE.
#endif
    struct is_prefix_of;
  }


  /**
   * \interface prefix_of<>
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \copydetails internal::is_prefix_of<>.
   * \details For example, <code>prefix_of<Coefficients<Axis>, Coefficients<Axis, angle::Radians>></code> returns <code>true</code>.
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
  struct MatrixTraits;

  template<typename T>
  struct MatrixTraits {};

  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct MatrixTraits<T> : public MatrixTraits<std::decay_t<T>> {};
#else
  template<typename T, typename Enable = void> ///< \tparam Enable A dummy parameter for selection with SFINAE.
  struct MatrixTraits;

  template<typename T, typename Enable>
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
  struct DistributionTraits;

  template<typename T>
  struct DistributionTraits {};

  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct DistributionTraits<T> : DistributionTraits<std::decay_t<T>> {};
#else
  template<typename T, typename Enable = void> ///< \tparam Enable A dummy parameter for selection with SFINAE.
  struct DistributionTraits;

  template<typename T, typename Enable>
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
     * \brief A type trait testing whether T can be wrapped in a covariance.
     * \note: This class should be specialized for all appropriate matrix classes.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void> ///< \tparam Enable A dummy parameter for selection with SFINAE.
#endif
    struct is_covariance_base : std::false_type {};
  }

  /**
   * \interface covariance_base<>
   * \brief T is an acceptable base matrix for a covariance (including square_root_covariance).
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
     * \brief A type trait testing whether T is acceptable to be wrapped in a typed_matrix.
     * \note: This class should be specialized for all appropriate matrix classes.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void> ///< \tparam Enable A dummy parameter for selection with SFINAE.
#endif
    struct is_typed_matrix_base : std::false_type {};
  }

  /**
   * \interface typed_matrix_base<>
   * \brief T is an acceptable base for a general typed matrix (e.g., matrix, mean, or euclidean_mean)
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
