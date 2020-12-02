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
    /// \tparam Enable A dummy parameter for selection with SFINAE.
    template<typename T, typename U, typename Enable = void>
#endif
    struct is_equivalent_to;
  }


  /**
   * \brief T is equivalent to U, where T and U are sets of coefficients.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - Coefficient<Ts...> is equivalent to Coefficient<Us...>, if each Ts is equivalent to its respective Us.
   * - Coefficient<T> is equivalent to T, and vice versa.
   * Example: \code equivalent_to<Axis, Coefficients<Axis>> \endcode returns <code>true</code>.
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
    /// \tparam Enable A dummy parameter for selection with SFINAE.
    template<typename T, typename U, typename Enable = void>
#endif
    struct is_prefix_of;
  }


  /**
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of Coefficients<C, Cs...> for any coefficients Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * Coefficients<> is a prefix of any set of coefficients.
   * Example, \code prefix_of<Coefficients<Axis>, Coefficients<Axis, angle::Radians>> \endcode returns <code>true</code>.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of = internal::is_prefix_of<T, U>::value;
#else
  inline constexpr bool prefix_of = internal::is_prefix_of<T, U>::value;
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
    template<typename T, typename Enable = void>
#endif
    struct is_covariance_nestable : std::false_type {};
  }

  /**
   * \brief T is an acceptable nested matrix for a covariance (including square_root_covariance).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
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
    template<typename T, typename Enable = void>
#endif
    struct is_typed_matrix_nestable : std::false_type {};
  }

  /**
   * \brief Specifies a type that is nestable in a general typed matrix (e.g., matrix, mean, or euclidean_mean)
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix_nestable = internal::is_typed_matrix_nestable<std::decay_t<T>>::value;
#else
  inline constexpr bool typed_matrix_nestable = internal::is_typed_matrix_nestable<std::decay_t<T>>::value;
#endif


  // ------------------------------------ //
  //   MatrixTraits, DistributionTraits   //
  // ------------------------------------ //

  /**
   * \internal
   * \brief A type trait class for any matrix T.
   * \details This class includes key information about a matrix or matrix expression, such as its dimensions,
   * coefficient types, etc.
   * <table class = "memberdecls">
   * <tr class="heading"><td colspan="2"><h2 class="groupheader">Static Public Attributes</h2></td></tr>
   * <tr><td class="memItemLeft" align="right" valign="top">static constexpr std::size_t&nbsp;</td>
   * <td id="afwtraitsdimension" class="memItemRight" valign="bottom"><b>dimension</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">The number of rows of the matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * <tr><td class="memItemLeft" align="right" valign="top">static constexpr std::size_t&nbsp;</td>
   * <td id="afwtraitscolumns" class="memItemRight" valign="bottom"><b>columns</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td>
   * <td class="mdescRight">The number fo columns in the matrix. <br /></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * </table>
   *
   * <table class = "memberdecls">
   * <tr class="heading"><td colspan="2"><h2 class="groupheader">Public Aliases</h2></td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>Scalar</b></td></tr>
   * <tr><td id="afwtraitsScalar" class="mdescLeft">&nbsp;</td><td class="mdescRight">Scalar type of T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>NestedMatrix</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (optional) If T has a nested matrix, this is an alias for that nested matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>SelfContained</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * A \ref self_contained matrix equivalent to T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td id="afwtraitsRC" class="memItemRight" valign="bottom"><b>RowCoefficients</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * The coefficient types associated with the rows of T.
   * This is only applicable for matrices with typed coefficients.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td id="afwtraitsCC" class="memItemRight" valign="bottom"><b>ColumnCoefficients</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * The coefficient types associated with the columns of T.
   * This is only applicable for matrices with typed coefficients.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *  <tr><td class="memTemplParams" colspan="2">
   * template&lt;std::size_t rows = <a href="afwtraitsdimension">dimension</a>,
   * std::size_t cols = <a href="afwtraitscolumns">columns</a>,
   * typename S = <a href="afwtraitsScalar">Scalar</a>&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>NativeMatrix</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * A writable, native matrix type equivalent in size and shape to this matrix by default.
   * Alternatively, you can specify the <code>rows</code>, <code>columns</code>, and scalar type <code>S</code> type
   * of the new matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref TriangleType storage_triangle = \ref TriangleType::lower,
   * std::size_t size = <a href="afwtraitsdimension">dimension</a>,
   * typename S = <a href="afwtraitsScalar">Scalar</a>&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>SelfAdjointBaseType</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (Available if T is a native matrix.) A writable, native self-adjoint matrix type equivalent to T.
   * Alternatively, you can specify the <code>storage_triangle</code> (upper, lower, diagonal) where the coefficients
   * are stored, the <code>size</code> of the matrix, and the scalar type <code>S</code> type
   * (integral or floating-point) of the new matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref TriangleType triangle_type = \ref TriangleType::lower,
   * std::size_t size = <a href="afwtraitsdimension">dimension</a>,
   * typename S = <a href="afwtraitsScalar">Scalar</a>&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>TriangularBaseType</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (Available if T is a native matrix.) A writable, native triangular matrix type equivalent to T.
   * Alternatively, you can specify the <code>triangle_type</code> (upper, lower, diagonal) of the triangular matrix,
   * the <code>size</code> of the matrix, and the scalar type <code>S</code> type (integral or floating-point)
   * of the new matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;std::size_t size = <a href="afwtraitsdimension">dimension</a>,
   * typename S = <a href="afwtraitsScalar">Scalar</a>&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>DiagonalBaseType</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (Available if T is a native matrix.) A writable, native diagonal matrix type equivalent to T.
   * Alternatively, you can specify the <code>size</code> of the matrix and
   * the scalar type <code>S</code> type (integral or floating-point)
   * of the new matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">template&lt;typename T&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>MatrixBaseType</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * A native base type for any class Derived for which T is a nested matrix class.
   * This is the mechanism by which new matrix types can inherit from a base class of the matrix library.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * </table>
   *
   * <table class = "memberdecls">
   * <tr class="heading"><td colspan="2"><h2 class="groupheader">Static Public Member Functions</h2></td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>zero</b> ()</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make a matrix of type T with only zero coefficients.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>identity</b> ()</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make an identity matrix based on T. The resulting type will be a
   * square matrix of size <a href="afwtraitsdimension">dimension</a>.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">template&lt;typename Arg&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>make</b> (Arg&& arg) noexcept</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make a matrix of type T from a native matrix of type Arg.
   * It might have a different size and shape.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref OpenKalman::coefficients "coefficients" RC = <a href="afwtraitsRC">RowCoefficients</a>,
   * \ref OpenKalman::coefficients "coefficients" CC = <a href="afwtraitsCC">ColumnCoefficients</a>,
   * \ref typed_matrix_nestable Arg&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>make</b> (Arg&& arg) noexcept</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (Available if T is a \ref typed_matrix.) Make a self-contained typed matrix based on T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref OpenKalman::coefficients "coefficients" C = <a href="afwtraitsRC">RowCoefficients</a>,
   * \ref covariance_nestable Arg&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>make</b> (Arg&& arg) noexcept</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (Available if T is a \ref covariance.) Make a self-contained covariance matrix based on T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;std::convertible_to&lt;const <a href="afwtraitsScalar">Scalar</a>&gt; ... Args&gt;
   * requires (sizeof...(Args) == <a href="afwtraitsdimension">dimension</a> * <a href="afwtraitscolumns">columns</a>)
   * </td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>make</b> (const Args ... args) noexcept</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (If T is a native matrix.) Make a self-contained native matrix from a list of coefficients
   * in row-major order.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * </table>
   * \tparam T The matrix type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  /// \tparam Enable A dummy parameter for selection with SFINAE.
  template<typename T, typename Enable = void>
#endif
  struct MatrixTraits {};


#ifdef __cpp_concepts
  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct MatrixTraits<T> : public MatrixTraits<std::decay_t<T>> {};
#else
  template<typename T>
  struct MatrixTraits<T&> : MatrixTraits<T> {};

  template<typename T>
  struct MatrixTraits<T&&> : MatrixTraits<T> {};

  template<typename T>
  struct MatrixTraits<const T> : MatrixTraits<T> {};
#endif


  /**
   * \brief A type trait class for any distribution T.
   * \details This class includes key information about a matrix or matrix expression, such as its dimensions,
   * coefficient types, etc.
   * <table class = "memberdecls">
   * <tr class="heading"><td colspan="2"><h2 class="groupheader">Static Public Attributes</h2></td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">static constexpr std::size_t&nbsp;</td>
   * <td id="afwtraitsDdimension" class="memItemRight" valign="bottom"><b>dimension</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">The number of rows of the matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * </table>
   *
   * <table class = "memberdecls">
   * <tr class="heading"><td colspan="2"><h2 class="groupheader">Public Aliases</h2></td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>Scalar</b></td></tr>
   * <tr><td id="afwtraitsDScalar" class="mdescLeft">&nbsp;</td><td class="mdescRight">Scalar type of T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>SelfContained</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * A \ref self_contained distribution equivalent to T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td id="afwtraitsDRC" class="memItemRight" valign="bottom"><b>%Coefficients</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * The types of \ref OpenKalman::coefficients "coefficients" associated with the
   * mean and covariance of T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>random_number_engine</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * The random number engine associated with generating samples within distribution T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * </table>
   *
   * <table class = "memberdecls">
   * <tr class="heading"><td colspan="2"><h2 class="groupheader">Static Public Member Functions</h2></td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>zero</b> ()</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make a distribution of type T in which the mean and covariance matrix have only zero coefficients.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memItemRight" valign="bottom"><b>normal</b> ()</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make a normal distribution based on T. It will have zero mean and an identity covariance matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref OpenKalman::coefficients "coefficients" C = <a href="afwtraitsDRC">Coefficients</a>,
   * \ref mean M, \ref OpenKalman::covariance "covariance" Cov&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>make</b> (M&& mean, Cov&& covariance) noexcept</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make a self-contained distribution based on T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref OpenKalman::coefficients "coefficients" C = <a href="afwtraitsDRC">Coefficients</a>,
   * \ref mean M, \ref OpenKalman::covariance "covariance" Cov&gt; requires (MatrixTraits&lt;M&gt;::columns == 1) and
   * (MatrixTraits&lt;Mean&gt;::dimension == MatrixTraits&lt;Cov&gt;::dimension)</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">static auto&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>make</b> (M&& mean, Cov&& covariance) noexcept</td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * Make a self-contained distribution based on T.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * </table>
   * \sa MatrixTraits
   * \tparam T The distribution type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  /// \tparam Enable A dummy parameter for selection with SFINAE.
  template<typename T, typename Enable = void>
#endif
  struct DistributionTraits {};


#ifdef __cpp_concepts
  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct DistributionTraits<T> : DistributionTraits<std::decay_t<T>> {};
#else
  template<typename T>
  struct DistributionTraits<T&> : DistributionTraits<T> {};

  template<typename T>
  struct DistributionTraits<T&&> : DistributionTraits<T> {};

  template<typename T>
  struct DistributionTraits<const T> : DistributionTraits<T> {};
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_TRAITS_HPP
