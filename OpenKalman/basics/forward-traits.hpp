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
 * \file
 * \brief Forward declarations for traits relating to OpenKalman or native matrix types.
 */

#ifndef OPENKALMAN_FORWARD_TRAITS_HPP
#define OPENKALMAN_FORWARD_TRAITS_HPP

#include <type_traits>


/**
 * \brief The root namespace for OpenKalman.
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

  }


  /**
   * \brief T is a group of atomic or composite coefficients.
   * \details Atomic coefficient groups are %coefficients or groups of %coefficients that function as a unit,
   * and cannot be separated. They may be combined into composite coefficients by passing them as template
   * parameters to Coefficients. These include Axis, Distance, Angle, Inclination, Polar, and Spherical.
   *
   * Composite coefficients are specializations of the class Coefficients, which has the purpose of grouping
   * other atomic or composite coefficients. Composite coefficients can, themselves, comprise groups of other
   * composite components. Composite coefficients are of the form Coefficients<Cs...>.
   *
   * Examples of coefficients:
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
   * \brief T is equivalent to U, where T and U are sets of coefficients.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - Coefficient<Ts...> is equivalent to Coefficient<Us...>, if each Ts is equivalent to its respective Us.
   * - Coefficient<T> is equivalent to T, and vice versa.
   * Example: <code>equivalent_to<Axis, Coefficients<Axis>></code> returns <code>true</code>.
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
     * \brief Type trait testing whether T (a set of coefficients) is a prefix of U.
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
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of Coefficients<C, Cs...> for any coefficients Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * Coefficients<> is a prefix of any set of coefficients.
   * Example, <code>prefix_of<Coefficients<Axis>, Coefficients<Axis, angle::Radians>></code> returns <code>true</code>.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of = internal::is_prefix_of<T, U>::value;
#else
  inline constexpr bool prefix_of = internal::is_prefix_of<T, U>::value;
#endif


  // ------------------------------------ //
  //   MatrixTraits, DistributionTraits   //
  // ------------------------------------ //

  /**
   * \brief A type trait class for any matrix T.
   * \details This class includes key information about a matrix or matrix expression, such as its dimensions,
   * coefficient types, etc.
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
   * \typedef MatrixTraits<T>::NestedMatrix
   * \brief The type of the nested matrix.
   *//**
   * \typedef MatrixTraits<T>::Scalar
   * \brief The scalar type.
   *//**
   * \variable static constexpr std::size_t MatrixTraits<T>::dimension
   * \brief The number of rows in the matrix.
   *//**
   * \variable static constexpr std::size_t MatrixTraits<T>::columns
   * \brief The number of columns.
   *//**
   * \typedef MatrixTraits<T>::Scalar
   * \brief The scalar type.
   *//**
   * \typedef MatrixTraits<T>::NativeMatrix<std::size_t rows, std::size_t cols, typename Scalar>
   * \brief A writable, native matrix type equivalent to this matrix.
   * \tparam rows The number of rows.
   * \tparam cols The number of columns.
   * \tparam Scalar The scalar type (integral or floating-point) for the new native matrix.
   *//**
   * \typedef MatrixTraits<T>::SelfAdjointBaseType<TriangleType storage_triangle, std::size_t dim, typename Scalar>
   * \brief A writable, native self-adjoint matrix type equivalent to this matrix.
   * \details This should be defined for any native interface types.
   * \tparam storage_triangle The triangle type (upper, lower) where the coefficients are stored.
   * \tparam dim The number of rows and columns.
   * \tparam Scalar The scalar type (integral or floating-point) for the new native matrix.
   *//**
   * \typedef MatrixTraits<T>::TriangularBaseType<TriangleType triangle_type, std::size_t dim, typename Scalar>
   * \brief A writable, native self-adjoint matrix type equivalent to this matrix.
   * \details This should be defined for any native interface types.
   * \tparam triangle_type The triangle type.
   * \tparam dim The number of rows and columns.
   * \tparam Scalar The scalar type (integral or floating-point) for the new native matrix.
   *//**
   * \typedef MatrixTraits<T>::SelfContained
   * \brief A self_contained type equivalent to this matrix.
   *//**
   * \typedef MatrixTraits<T>::Coefficients
   * \brief The coefficient types associated with this matrix.
   * \details Only applicable for matrices with typed coefficients.
   *//**
   * \typedef MatrixTraits<T>::RowCoefficients
   * \brief The coefficient types associated with the rows.
   * \details Only applicable for matrices with typed coefficients. If this is defined, then ColumnCoefficients
   * should also be defined, and Coefficients should ''not'' be defined.
   *//**
   * \typedef MatrixTraits<T>::ColumnCoefficients
   * \brief The coefficient types associated with this columns.
   * \details Only applicable for matrices with typed coefficients. If this is defined, then RowCoefficients
   * should also be defined, and Coefficients should ''not'' be defined.
   *//**
   * \fn static auto MatrixTraits<T>::make<coefficients RC, coefficients CC, typed_matrix_nestable Arg>(Arg&& arg)
   * \brief Makes a self-contained typed matrix based on this type.
   * \details This is only defined for matrices with typed coefficients.
   * \tparam RC The row coefficients for the new matrix.
   * \tparam CC The column coefficients for the new matrix.
   * \tparam Arg a matrix type, which is nestable in a typed matrix, on which the new matrix is based.
   *//**
   * \fn static auto MatrixTraits<T>::zero()
   * \brief Makes a zero matrix based on this type.
   * \details All the coefficients are zero.
   *//**
   * \fn static auto MatrixTraits<T>::identity()
   * \brief Makes an identity matrix based on this type.
   * \details The resulting type will be a square matrix, of dimension MatrixTraits<>::dimension.
   *//**
   * \typedef MatrixTraits<T>::MatrixBaseType<typename Derived>
   * \brief A native base type for any class Derived for which T is a nested matrix class.
   * \details This is the mechanism by which new matrix types can inherit from a base class of the matrix library.
   * \tparam Derived The type using T as a nested class.
   */



  /**
   * \brief A type trait class for any distribution T.
   * \sa MatrixTraits
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


  // ------------------------- //
  //    covariance_nestable    //
  // ------------------------- //

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
    struct is_covariance_nestable : std::false_type {};
  }

  /**
   * \brief T is an acceptable nested matrix for a covariance (including square_root_covariance).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance_nestable = internal::is_covariance_nestable<std::decay_t<T>>::value;
#else
  inline constexpr bool covariance_nestable = internal::is_covariance_nestable<std::decay_t<T>>::value;
#endif


  // ----------------------- //
  //    typed_matrix_nestable    //
  // ----------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether T is acceptable to be nested in a typed_matrix.
     * \note: This class should be specialized for all appropriate matrix classes.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void> ///< \tparam Enable A dummy parameter for selection with SFINAE.
#endif
    struct is_typed_matrix_nestable : std::false_type {};
  }

  /**
   * \brief Specifies a type that is nestable in a general typed matrix (e.g., matrix, mean, or euclidean_mean)
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix_nestable = internal::is_typed_matrix_nestable<std::decay_t<T>>::value;
#else
  inline constexpr bool typed_matrix_nestable = internal::is_typed_matrix_nestable<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_TRAITS_HPP
