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
 * \brief Forward declarations for traits relating to OpenKalman or native matrix types.
 */

#ifndef OPENKALMAN_FORWARD_TRAITS_HPP
#define OPENKALMAN_FORWARD_TRAITS_HPP

#include <type_traits>


/**
 * \namespace OpenKalman
 * \brief The root namespace for OpenKalman.
 *
 * \internal
 * \namespace OpenKalman::internal
 * \brief Namespace for internal definitions, not intended for use outside of OpenKalman development.
 */


namespace OpenKalman
{
  // ---------------- //
  //   Coefficients   //
  // ---------------- //

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
   * <b>Examples</b>:
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
    template<typename T, typename U, typename = void>
#endif
    struct is_equivalent_to;
  }


  /**
   * \brief T is equivalent to U, where T and U are sets of coefficients.
   * \details Sets of coefficients are equivalent if they are treated functionally the same.
   * - Any coefficient or group of coefficients is equivalent to itself.
   * - Coefficient<Ts...> is equivalent to Coefficient<Us...>, if each Ts is equivalent to its respective Us.
   * - Coefficient<T> is equivalent to T, and vice versa.
   * \par Example:
   * <code>equivalent_to&lt;Axis, Coefficients&lt;Axis&gt;&gt;
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
    template<typename T, typename U, typename = void>
#endif
    struct is_prefix_of;
  }


  /**
   * \brief T is a prefix of U, where T and U are sets of coefficients.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of Coefficients<C, Cs...> for any coefficients Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * Coefficients<> is a prefix of any set of coefficients.
   * \par Example:
   * <code>prefix_of&lt;Coefficients&lt;Axis&gt;, Coefficients&lt;Axis, angle::Radians&gt;&gt;
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
    template<typename T, typename = void>
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
    template<typename T, typename = void>
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
   * <td class="memItemRight" valign="bottom"><b>SelfContainedFrom</b></td></tr>
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
   * <td class="memTemplItemRight" valign="bottom"><b>NativeMatrixFrom</b></td></tr>
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
   * <td class="memTemplItemRight" valign="bottom"><b>SelfAdjointMatrixFrom</b></td></tr>
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
   * <td class="memTemplItemRight" valign="bottom"><b>TriangularMatrixFrom</b></td></tr>
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
   * <td class="memTemplItemRight" valign="bottom"><b>DiagonalMatrixFrom</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td><td class="mdescRight">
   * (Available if T is a native matrix.) A writable, native diagonal matrix type equivalent to T.
   * Alternatively, you can specify the <code>size</code> of the matrix and
   * the scalar type <code>S</code> type (integral or floating-point)
   * of the new matrix.<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   *   <tr><td class="memTemplParams" colspan="2">template&lt;typename T&gt;</td></tr>
   * <tr><td class="memTemplItemLeft" align="right" valign="top">using&nbsp;</td>
   * <td class="memTemplItemRight" valign="bottom"><b>MatrixBaseFrom</b></td></tr>
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
  template<typename T, typename = void>
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
   * \internal
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
   * <td class="memItemRight" valign="bottom"><b>SelfContainedFrom</b></td></tr>
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
   * \ref mean M, \ref OpenKalman::covariance "covariance" Cov&gt; requires column_vector&lt;M&gt; and
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
  template<typename T, typename = void>
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


  // ----------------- //
  //  nested_matrix_t  //
  // ----------------- //

  /**
   * \brief An alias for a type's nested matrix, if it exists.
   * \details Only participates in overload resolution if the type has a nested matrix.
   * \tparam T A type that is a wrapper for a nested matrix.
   */
#ifdef __cpp_concepts
  template<typename T> requires (requires {typename MatrixTraits<T>::NestedMatrix;})
#else
  template<typename T, typename = typename MatrixTraits<T>::NestedMatrix>
#endif
  using nested_matrix_t = typename MatrixTraits<T>::NestedMatrix;


  // ----------------- //
  //  native_matrix_t  //
  // ----------------- //

  /**
   * \brief An alias for a self-contained native matrix, based on and equivalent to parameter T.
   * \tparam T The type from which the native matrix is derived.
   * \tparam rows Number of rows in the native matrix (defaults to the number of rows in T).
   * \tparam cols Number of columns in the native matrix (defaults to the number of columns in T).
   * \tparam Scalar Scalar type of the matrix (defaults to the Scalar type of T).
   */
  template<typename T, std::size_t rows = MatrixTraits<T>::dimension, std::size_t cols = MatrixTraits<T>::columns,
    typename Scalar = typename MatrixTraits<T>::Scalar>
  using native_matrix_t = typename MatrixTraits<T>::template NativeMatrixFrom<rows, cols, Scalar>;


  /**
   * \brief Make a self-contained, native matrix based on the shape of M from a list of coefficients in row-major order.
   */
#ifdef __cpp_concepts
  template<typename M, std::convertible_to<typename MatrixTraits<M>::Scalar> ... Args> requires
  (sizeof...(Args) == MatrixTraits<M>::dimension * MatrixTraits<M>::columns)
#else
  template<typename M, typename ... Args, std::enable_if_t<
    (std::is_convertible_v<Args, typename MatrixTraits<M>::Scalar> and ...) and
    (sizeof...(Args) == MatrixTraits<M>::dimension * MatrixTraits<M>::columns), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    return MatrixTraits<native_matrix_t<M>>::make(static_cast<const typename MatrixTraits<M>::Scalar>(args)...);
  }


  // -------------- //
  //  TriangleType  //
  // -------------- //

  /**
   * \brief The type of a triangular matrix, either lower, upper, or diagonal.
   */
  enum struct TriangleType {
    lower, ///< The lower-left triangle.
    upper, ///< The upper-right triangle.
    diagonal ///< The diagonal elements of the matrix.
  };


  // ---------------- //
  //  self_contained  //
  // ---------------- //

  namespace internal
  {
    /**
     * \internal
     * \details Type trait testing whether T is self-contained (i.e., can be the return value of a function).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_self_contained : std::false_type {};
  }


  /**
   * \brief Specifies that a type is a self-contained matrix or expression.
   * \details A value is self-contained if it can be created in a function and returned as the result.
   * An OpenKalman matrix type is self-contained if it is not an lvalue reference and its wrapped native matrix
   * is self-contained.
   * The matrix library interface will specify which native matrices and expressions are self-contained.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T, typename...Ts>
#ifdef __cpp_concepts
  concept self_contained =
#else
    constexpr bool self_contained =
#endif
    internal::is_self_contained<std::decay_t<T>>::value and (not std::is_lvalue_reference_v<T>);


  // ------------- //
  //  zero_matrix  //
  // ------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is a zero matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_zero_matrix : std::false_type {};
  }


  /**
   * \brief Specifies that a type is a zero matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero_matrix = internal::is_zero_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool zero_matrix = internal::is_zero_matrix<std::decay_t<T>>::value;
#endif


  // ----------------- //
  //  identity_matrix  //
  // ----------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is an identity matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_identity_matrix : std::false_type {};
  }

  /**
   * \brief Specifies that a type is an identity matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix = internal::is_identity_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool identity_matrix = internal::is_identity_matrix<std::decay_t<T>>::value;
#endif


  // ------------------- //
  //  one_by_one_matrix  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_1by1 : std::false_type {};

    template<typename T>
    struct is_1by1<T, std::enable_if_t<(MatrixTraits<T>::dimension == 1) and (MatrixTraits<T>::columns == 1)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a one-by-one matrix (i.e., one row and one column).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept one_by_one_matrix = (MatrixTraits<T>::dimension == 1) and (MatrixTraits<T>::columns == 1);
#else
  inline constexpr bool one_by_one_matrix = detail::is_1by1<T>::value;
#endif


  // --------------- //
  //  square_matrix  //
  // --------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_square_matrix : std::false_type {};
  }

  /**
   * \brief Specifies that a matrix is square (i.e., has the same number and type of rows and column).
   * \details If T is a \ref typed_matrix, the row coefficients must also be \ref equivalent_to the column coefficients.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept square_matrix = internal::is_square_matrix<T>::value;
#else
  inline constexpr bool square_matrix = internal::is_square_matrix<T>::value;
#endif


  // -------------------- //
  //  diagonal_matrix  //
  // -------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is a diagonal matrix
     * \note This excludes zero_matrix, identity_matrix, or one_by_one_matrix.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_diagonal_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_diag_matrix_impl : std::false_type {};

    template<typename T>
    struct is_diag_matrix_impl<T, std::enable_if_t<internal::is_diagonal_matrix<std::decay_t<T>>::value or
      (zero_matrix<T> and square_matrix<T>) or identity_matrix<T> or one_by_one_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a diagonal matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept diagonal_matrix = internal::is_diagonal_matrix<std::decay_t<T>>::value or
    (zero_matrix<T> and square_matrix<T>) or identity_matrix<T> or one_by_one_matrix<T>;
#else
  template<typename T>
  inline constexpr bool diagonal_matrix = detail::is_diag_matrix_impl<T>::value;
#endif


  // ------------------------ //
  //  is_self_adjoint_matrix  //
  // ------------------------ //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is a self-adjoint matrix (other than diagonal_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_self_adjoint_matrix : std::false_type {};
  }


  /**
   * \brief Specifies that a type is a self-adjoint matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept self_adjoint_matrix = internal::is_self_adjoint_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#else
  template<typename T>
  inline constexpr bool self_adjoint_matrix =
    internal::is_self_adjoint_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#endif


  // ------------------------- //
  //  lower_triangular_matrix  //
  // ------------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is a lower-triangular matrix (other than diagonal_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_lower_triangular_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_lt_matrix_impl : std::false_type {};

    template<typename T>
    struct is_lt_matrix_impl<T, std::enable_if_t<
      internal::is_lower_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a lower-triangular matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept lower_triangular_matrix = internal::is_lower_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#else
  template<typename T>
  inline constexpr bool lower_triangular_matrix = detail::is_lt_matrix_impl<T>::value;
#endif


  // ------------------------- //
  //  upper_triangular_matrix  //
  // ------------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether an object is an upper-triangular matrix (other than diagonal_matrix).
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_upper_triangular_matrix : std::false_type {};
  }


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_ut_matrix_impl : std::false_type {};

    template<typename T>
    struct is_ut_matrix_impl<T, std::enable_if_t<
      internal::is_upper_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is an upper-triangular matrix.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept upper_triangular_matrix = internal::is_upper_triangular_matrix<std::decay_t<T>>::value or diagonal_matrix<T>;
#else
  template<typename T>
  inline constexpr bool upper_triangular_matrix = detail::is_ut_matrix_impl<T>::value;
#endif


  // ------------------- //
  //  triangular_matrix  //
  // ------------------- //

  /**
   * \brief Specifies that a type is a triangular matrix (upper or lower).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept triangular_matrix = lower_triangular_matrix<T> or upper_triangular_matrix<T>;
#else
  template<typename T>
  inline constexpr bool triangular_matrix = lower_triangular_matrix<T> or upper_triangular_matrix<T>;
#endif


  // --------------------------------- //
  //  internal::same_triangle_type_as  //
  // --------------------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief Specifies that two types have the same triangular type (upper or lower).
     * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
     */
    template<typename T, typename U>
#ifdef __cpp_concepts
    concept same_triangle_type_as =
      (upper_triangular_matrix<T> and upper_triangular_matrix<U>) or
      (lower_triangular_matrix<T> and lower_triangular_matrix<U>);
#else
    inline constexpr bool same_triangle_type_as =
      (upper_triangular_matrix<T> and upper_triangular_matrix<U>) or
        (lower_triangular_matrix<T> and lower_triangular_matrix<U>);
#endif
  }


  // --------------- //
  //  cholesky_form  //
  // --------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A type trait testing whether the nested matrix is a Cholesky square root.
     */
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct is_cholesky_form : std::false_type {};
  }


  /**
   * \brief Specifies that a type has a nested native matrix that is a Cholesky square root.
   * \details If this is true, then nested_matrix_t<T> is true.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept cholesky_form = internal::is_cholesky_form<std::decay_t<T>>::value;
#else
  inline constexpr bool cholesky_form = internal::is_cholesky_form<std::decay_t<T>>::value;
#endif


  // --------------------- //
  //  is_element_gettable  //
  // --------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether an object has elements that can be retrieved with N indices.
     * \note This should be specialized for all matrix types usable with OpenKalman.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N>
    struct is_element_gettable : std::false_type {};
#else
    template<typename T, std::size_t N, typename = void>
    struct is_element_gettable : std::false_type {};
#endif
  }


  /**
   * \brief Specifies that a type has elements that can be retrieved with N number of indices.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept element_gettable = internal::is_element_gettable<std::decay_t<T>, N>::value;
#else
  inline constexpr bool element_gettable = internal::is_element_gettable<std::decay_t<T>, N>::value;
#endif


  // --------------------- //
  //  is_element_settable  //
  // --------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief Type trait testing whether an object has elements that can be set with N indices.
     * \note This should be specialized for all matrix types usable with OpenKalman.
     */
#ifdef __cpp_concepts
    template<typename T, std::size_t N>
    struct is_element_settable : std::false_type {};
#else
    template<typename T, std::size_t N, typename = void>
    struct is_element_settable : std::false_type {};
#endif
  }

  /**
   * \brief Specifies that a type has elements that can be set with N number of indices.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept element_settable = internal::is_element_settable<T, N>::value and
    (not std::is_const_v<std::remove_reference_t<T>>);
#else
  inline constexpr bool element_settable = internal::is_element_settable<T, N>::value and
    (not std::is_const_v<std::remove_reference_t<T>>);
#endif


  // ------------ //
  //  modifiable  //
  // ------------ //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
    struct is_modifiable : std::true_type {};
#else
    template<typename T, typename U, typename = void>
    struct is_modifiable : std::true_type {};
#endif


    // Custom modifiability parameter that can be defined in the native matrix ecosystem.
#ifdef __cpp_concepts
    template<typename T, typename U>
    struct is_modifiable_native : std::true_type {};
#else
    template<typename T, typename U, typename = void>
    struct is_modifiable_native : std::true_type {};
#endif

  } // namespace internal


  /**
   * \internal
   * \brief Specifies that U is not obviously incompatible with T, such that assigning U to T might be possible.
   * \details The result is true unless there is an incompatibility of some kind that would prevent assignment.
   * Examples of such incompatibility are if T is constant or has a nested constant type, if T and U have a
   * different shape or scalar type, or if T and U differ as to being self-adjoint, triangular, diagonal,
   * zero, or identity. Even if this concept is true, a compile-time error is still possible.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept modifiable = internal::is_modifiable<T, U>::value and internal::is_modifiable_native<T, U>::value;
#else
  inline constexpr bool modifiable =
    internal::is_modifiable<T, U>::value and internal::is_modifiable_native<T, U>::value;
#endif


} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_TRAITS_HPP
