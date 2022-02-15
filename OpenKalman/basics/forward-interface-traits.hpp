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
 * \brief Forward declarations for interface traits, which must be defined for all matrices used in OpenKalman.
 */

#ifndef OPENKALMAN_FORWARD_INTERFACE_TRAITS_HPP
#define OPENKALMAN_FORWARD_INTERFACE_TRAITS_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman
{
  /**
   * \brief A constant indicating that the relevant dimension of a matrix is dynamic.
   * \sa std::dynamic_extent
   */
  static constexpr std::size_t dynamic_extent = 0; //std::numeric_limits<std::size_t>::max();
}


/**
 * \brief The root namespace for OpenKalman interface types.
 */
namespace OpenKalman::interface
{

  /**
   * \internal
   * \brief Type row extent of a matrix, expression, or array.
   * \details The interface must define the following:
   *   - Static member <code>static constexpr std::size_t value</code> representing
   *   the row extent. RowExtentOf may, e.g., derive from <code>std::integral_constant<std::size_t, value></code>.
   *   - static member function <code>rows_at_runtime</code>, returning the number of rows in T, evaulated at runtime.
   * \note If the row extent is dynamic, then <code>value</code> should be \ref dynamic_extent.
   * \tparam T The matrix, expression, or array.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct RowExtentOf
  {
    /**
     * \tparam Arg A matrix of type T
     * \return The number of rows of T, evaluated at runtime.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, std::decay_t<T>>
#else
    template<typename Arg, std::enable_if_t<std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>, int> = 0>
#endif
    static constexpr std::size_t rows_at_runtime(Arg&& arg) = delete;


    /**
     * \var value
     * \brief The number of rows of T, evaluated at compile time.
     * \code
     *   static constexpr std::size_t value = 0;
     * \endcode
     */
  };


  /**
   * \internal
   * \brief Type column extent of a matrix, expression, or array.
   * \details The interface must define the following:
   *   - Static member <code>static constexpr std::size_t value</code> representing
   *   the column extent. RowExtentOf may, e.g., derive from <code>std::integral_constant<std::size_t, value></code>.
   *   - static member function <code>columns_at_runtime</code>, returning the number of columns in T, evaulated at
   *   runtime.
   * \note If the column extent is dynamic, then <code>value</code> should be \ref dynamic_extent.
   * \tparam T The matrix, expression, or array.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ColumnExtentOf
  {
    /**
     * \tparam Arg A matrix of type T
     * \return The number of rows of T, evaluated at runtime.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, std::decay_t<T>>
#else
    template<typename Arg, std::enable_if_t<std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>, int> = 0>
#endif
    static constexpr std::size_t columns_at_runtime(Arg&& arg) = delete;


    /**
     * \var value
     * \brief The number of columns of T, evaluated at compile time.
     * \code
     *   static constexpr std::size_t value = 0;
     * \endcode
     */
  };


  /**
   * \internal
   * \brief Type trait identifying the scalar type (e.g., double, int) of a matrix, expression, or array.
   * \details The interface must define a member alias <code>type</code> as the scalar type.
   * \tparam T The matrix, expression, or array.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ScalarTypeOf;


  /**
   * \internal
   * \brief Interface to a dense, writable, self-contained matrix or array that is equivalent to T.
   * \details The resulting type is equivalent to T, but may be have a specified shape or scalar type. The interface
   * can set the size or scalar type of the resulting dense matrix based on the parameters (or if they are dynamic,
   * row_extent and column_extent can be set to \ref dynamic_extent).
   * The interface must define the following:
   *   - member alias <code>type</code> as the equivalent matrix/array type;
   *   - static member function <code>make_default(...)</code>, taking 0, 1, or 2 runtime arguments for any
   *   row_extent or column_extent that are \ref dynamic_extent and returning a default, uninitialized, self-contained
   *   matrix/array of type T--for example, it takes 0 parameters if both row_extent and column_extent are fixed, and it
   *   takes 2 parameters (row and column) if both row_extent and column_extent are dynamic; and
   *   - static member function <code>convert</code> (<var>convert</var> : <var>T</var> -> <var>type</var>),
   *   which converts a matrix/array convertible to type <code>T</code> into a dense, writable matrix/array of type
   *   <code>type</code>.
   * Example of use in defining an interface:
   * \code
   *   template<MyMatrixExpression T, std::size_t row_extent, std::size_t column_extent, typename scalar_type>
   *   struct EquivalentDenseWritableMatrix<T, row_extent, column_extent, scalar_type>
   *   {
   *     using type = MyDenseMatrix<scalar_type, row_extent, column_extent>;
   *
   *     template<std::convertible_to<std::size_t>...extents> requires
   *       (sizeof...(extents) == (row_extent == dynamic_extent ? 1 : 0) + (column_extent == dynamic_extent ? 1 : 0))
   *     static type make_default(extents...e)
   *     {
   *       // in this example, MyDenseMatrix takes either two runtime extents (if dynamic) or zero (if fixed)
   *       if constexpr (row_extent == dynamic_extent)
   *       {
   *         if constexpr (column_extent == dynamic_extent)
   *           return type(e...);
   *         else
   *           return type(e..., column_extent);
   *       }
   *       else
   *       {
   *         if constexpr (column_extent == dynamic_extent)
   *           return type(row_extent, e...);
   *         else
   *           return type(); // fixed shape
   *       }
   *     }
   *
   *     template<typename Arg> requires
   *       (row_extent_of_v<Arg> == dynamic_extent or row_extent_of_v<Arg> == row_extent) and
   *       (column_extent_of_v<Arg> == dynamic_extent or column_extent_of_v<Arg> == column_extent) and
   *       (column_extent_of_v<Arg> == column_extent) and
   *       std::convertible_to<scalar_type_of_t<Arg>, scalar_type>
   *     static decltype(auto) convert(Arg&& arg)
   *     {
   *       if constexpr (std::is_same_v<std::decay_t<Arg>, type>)
   *       {
   *         return std::forward<Arg>(arg);
   *       }
   *       else
   *       {
   *         return type {std::forward<Arg>(arg)};
   *       }
   *     }
   *   };
   * \endcode
   * \tparam T Type upon which the dense matrix will be constructed
   * \tparam row_extent The specified row extent of the matrix (defaults to that of T)
   * \tparam column_extent The specified column extent of the matrix (defaults to that of T)
   * \tparam scalar_type The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T,
    std::size_t row_extent = RowExtentOf<std::decay_t<T>>::value,
    std::size_t column_extent = ColumnExtentOf<std::decay_t<T>>::value,
    typename scalar_type = typename ScalarTypeOf<std::decay_t<T>>::type>
#else
  template<typename T,
    std::size_t row_extent = RowExtentOf<std::decay_t<T>>::value,
    std::size_t column_extent = ColumnExtentOf<std::decay_t<T>>::value,
    typename scalar_type = typename ScalarTypeOf<std::decay_t<T>>::type,
    typename = void>
#endif
  struct EquivalentDenseWritableMatrix;

  /**
   * \todo Add an interface to an equivalent dense readable matrix.
   * \todo Add a custom Eigen expression that is simply a wrapper for any OpenKalman type, along with its own evaluator.
   */


  /**
   * \internal
   * \brief An interface to T's nested matrices or other dependencies, whether embedded in T or nested by reference.
   * \details The interface must define a <code>std::tuple</code> as member alias <code>type</code>, where the tuple
   * elements correspond to each dependent object for which Dependencies is also defined. Such dependent objects may
   * include nested matrices or any parameters (e.g., indices indicating a particular block within a matrix).
   * The tuple element should be an lvalue reference if it is stored as an lvalue reference in type T).
   * The interface may define the following:
   *   - a static boolean member <code>has_runtime_parameters</code> that indicates whether type T stores any internal
   *   runtime parameters;
   *   - a member alias <code>type</code>, which is a tuple of elements corresponding to each dependency (the tuple
   *   element should be an lvalue reference if it is stored in T as an lvalue reference, and each included type should
   *   also have its own instance of Dependencies defined for it);
   *   - static member function <code>get_nested_matrix</code> that returns one of the dependencies; and
   *   - static member function <code>convert_to_self_contained</code> that converts a matrix convertible to type T
   *   into a self-contained object (optional if <code>type</code> is an empty tuple).
   * \tparam T A matrix, array, expression, distribution, etc., that has dependencies
   * \sa self_contained, make_self_contained, equivalent_self_contained_t, equivalent_dense_writable_matrix,
   * self_contained_parameter
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct Dependencies
  {
    /**
     * \brief Indicates whether type T stores any internal runtime parameters.
     * \details An example of an internal runtime paramter might be indices for start locations, or sizes, for an
     * expression representing a block or sub-matrix within a matrix. If unknown, the value of <code>true</code> is
     * the safest and will prevent unintended dangling references.
     * \note If this is not defined, T will be treated as if it is defined and true.
     */
    static constexpr bool has_runtime_parameters = true;


    /**
     * \brief Gets the i-th dependency of T.
     * /detail There is no need to check the bounds of <code>i</code>, but they should be treated as following this
     * constraint:
     * /code
     *   requires (i < std::tuple_size_v<type>) and std::same_as<std::decay_t<Arg>, std::decay_t<T>>
     * /endcode
     * \note Defining this function is optional. Also, there is no need for the example constraints on i or Arg,
     * as OpenKalman::nested_matrix already enforces these constraints.
     * \tparam i Index of the dependency (0 for the 1st dependency, 1 for the 2nd, etc.).
     * \tparam Arg An object of type T
     * \return The i-th dependency of T
     * \sa OpenKalman::nested_matrix
     */
    template<std::size_t i, typename Arg>
    static decltype(auto) get_nested_matrix(Arg&& arg) = delete;


    /**
     * \brief Converts an object of type T into an equivalent, self-contained object (i.e., no external dependencies).
     * \detail The resulting type must be equivalent to T, including in shape and scalar type. But it must be
     * self-contained, so that it has external dependencies accessible only by lvalue references. The result must be
     * guaranteed to be returnable from a function without causing a dangling reference. If possible, this should
     * preserve the traits of T, such as whether it is a \ref triangular_matrix, \ref diagonal_matrix, or
     * \note Defining this function is optional. If not defined, the default behavior is to convert to the equivalent,
     * dense, writable matrix. Also, there is no need for the example constraint on Arg, as
     * OpenKalman::make_self_contained already enforces this constraint.
     * \ref zero_matrix.
     * \tparam Arg An object of type T
     * \return An equivalent self-contained version of T
     */
#ifdef __cpp_concepts
    template<typename Arg> requires std::same_as<std::decay_t<Arg>, std::decay_t<T>>
#else
    template<typename Arg, std::enable_if_t<std::is_same_v<std::decay_t<Arg>, std::decay_t<T>>, int> = 0>
#endif
    static decltype(auto) convert_to_self_contained(Arg&& arg) = delete;


    /**
     * \typedef type
     * \brief A tuple with elements corresponding to each dependent object.
     * \details If the object is linked within T by an lvalue reference, the element should be an lvalue reference.
     * Examples:
     * \code
     *   using type = std::tuple<>; //< T has no dependencies
     *   using type = std::tuple<Arg1, Arg2&>; //< T stores Arg1 and a reference to Arg2
     * \endcode
     * \note If this is not defined, T will be considered non-self-contained.
     */
  };


  /**
   * \brief If T is a constant matrix, this is an interface to that constant.
   * \details The interface must define static constexpr member <code>value</code> representing the constant.
   * The type of <code>value</code> must be convertible to <code>scalar_type_of<T></code>.
   * \note This need only be defined for matrices in which every element is a constant expression.
   * \tparam T
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct SingleConstant
  {
    /**
     * \var value
     * \brief The constant element of T, of a type convertible to <code>scalar_type_of<T></code>.
     * \details The following example indicates that every element of T is 0 (same scalar type as T):
     * \code
     *   static constexpr typename ScalarTypeOf<T>::type value = 0;
     * \endcode
     */
  };


  /**
   * \brief If T is a constant-diagonal matrix, this is an interface to that constant.
   * \details The interface must define static constexpr member <code>value</code> representing the constant.
   * The type of <code>value</code> must be convertible to <code>scalar_type_of<T></code>.
   * \note This need only be defined for diagonal matrices in which every diagonal element is a single constant.
   * \tparam T
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct SingleConstantDiagonal
  {
    /**
     * \var value
     * \brief The constant element of T, of a type convertible to <code>scalar_type_of<T></code>.
     * \details The following example indicates that every diagonal element of T is 1 (same scalar type as T), and every
     * non-diagonal element is 0:
     * \code
     *   static constexpr typename ScalarTypeOf<T>::type value = 1;
     * \endcode
     */
  };


  /**
   * \brief An interface to features for getting individual elements of matrix T using indices I... of type std::size_t.
   * \detail The interface may define static member function <code>get</code> with one or two indices. If
   * getting an element is not possible, leave <code>get</code> undefined.
   * \note OpenKalman only recognizes indices of type <code>std::size_t</code>.
   * \tparam I The indices (each of type std::size_t)
   */
#ifdef __cpp_concepts
  template<typename T, typename...I>
#else
  template<typename T, typename = void, typename...I>
#endif
  struct GetElement
  {
    static_assert((std::is_convertible_v<I, const std::size_t&> and ...));

    /// Get element at indices (i...) of matrix arg
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I...i) = delete;
  };


  /**
   * \brief An interface to features for setting individual elements of matrix T using indices I... of type std::size_t.
   * \detail The interface may define static member function <code>set</code> with one or two indices. If
   * setting an element is not possible, leave <code>set</code> undefined.
   * \note OpenKalman only recognizes indices of type <code>std::size_t</code>.
   * \tparam I The indices (each of type std::size_t)
   */
#ifdef __cpp_concepts
  template<typename T, typename...I>
#else
  template<typename T, typename = void, typename...I>
#endif
  struct SetElement
  {
    static_assert((std::is_convertible_v<I, const std::size_t&> and ...));

    /// Set element at indices (i...) of matrix arg to s.
    template<typename Arg>
    static void set(Arg& arg, const typename ScalarTypeOf<std::decay_t<Arg>>::type& s, I...i) = delete;
  };


  /**
   * \brief An interface to necessary element-wise operations on matrix T.
   * \tparam T
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ElementWiseOperations
  {
    /**
     * \brief Fold an operation across the elements of Arg
     * \detail BinaryFunction must be invocable with two values, the first an accumulator and the second of type
     * <code>scalar_type_of_t<Arg></code>. It returns the accumulated value. After each iteration, the result of the
     * operation is used as the accumulator for the next iteration.
     * \tparam BinaryFunction A binary function (e.g. std::plus, std::multiplies)
     * \tparam Accum An accumulator
     * \tparam Arg An object of type T
     * \tparam order The element order over which to perform the operation
     * \todo derive default order from element order of Arg.
     */
    template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
    static constexpr decltype(auto) fold(const BinaryFunction&, Accum&&, Arg&&) = delete;
  };


  /**
   * \brief An interface to necessary conversions on matrix T.
   * \tparam T
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct Conversions
  {
  };


  /**
   * \brief An interface to necessary linear algebra operations operable on matrix T.
   * \tparam T
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct LinearAlgebra
  {
    /**
     * \brief Take the conjugate of T
     * \tparam Arg An object of type T
     */
    template<typename Arg>
    static constexpr auto conjugate(Arg&&) = delete;


    /**
     * \brief Take the transpose of T
     * \tparam Arg An object of type T
     */
    template<typename Arg>
    static constexpr auto transpose(Arg&&) = delete;


    /**
     * \brief Take the adjoint of T
     * \tparam Arg An object of type T
     */
    template<typename Arg>
    static constexpr auto adjoint(Arg&&) = delete;


    /**
     * \brief Take the determinant of T
     * \tparam Arg An object of type T
     */
    template<typename Arg>
    static constexpr auto determinant(Arg&&) = delete;


    /**
     * \brief Take the trace of T
     * \tparam Arg An object of type T
     */
    template<typename Arg>
    static constexpr auto trace(Arg&&) = delete;


    /**
     * \brief Do a rank update on a native Eigen matrix, treating it as a self-adjoint matrix.
     * \details If A is not hermitian, the result will modify only the specified storage triangle. The contents of the
     * other elements outside the specified storage triangle are undefined.
     * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
     * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
     * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
     * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
     * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
     * \tparam A An object of type T, which is the matrix to be rank updated.
     * \tparam U The update vector or matrix.
     * \returns an updated native, writable matrix in hermitian form.
     */
    template<TriangleType t, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_self_adjoint(A&&, U&&, const Alpha) = delete;


    /**
     * \brief Do a rank update on a native Eigen matrix, treating it as a triangular matrix.
     * \details If A is not a triangular matrix, the result will modify only the specified triangle. The contents of
     * other elements outside the specified triangle are undefined.
     * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
     * returning the updated A.
     * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
     * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
     * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
     * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
     * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
     * \tparam A An object of type T, which is the matrix to be rank updated.
     * \tparam U The update vector or matrix.
     * \returns an updated native, writable matrix in triangular (or diagonal) form.
     */
    template<TriangleType t, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&&, U&&, const Alpha) = delete;

  };


} // namespace OpenKalman::interface


namespace OpenKalman
{

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
   * <td id="afwtraitsrows" class="memItemRight" valign="bottom"><b>rows</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td>
   * <td class="mdescRight">The number of rows of the matrix (or 0 if dynamic).<br/></td></tr>
   * <tr><td class="memSeparator" colspan="2">&nbsp;</td></tr>
   * <tr><td class="memItemLeft" align="right" valign="top">static constexpr std::size_t&nbsp;</td>
   * <td id="afwtraitscolumns" class="memItemRight" valign="bottom"><b>columns</b></td></tr>
   * <tr><td class="mdescLeft">&nbsp;</td>
   * <td class="mdescRight">The number of columns in the matrix (or 0 if dynamic). <br /></td></tr>
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
   *   <tr><td class="memTemplParams" colspan="2">
   * template&lt;\ref TriangleType storage_triangle = \ref TriangleType::lower,
   * std::size_t size = <a href="afwtraitsrows">rows</a>,
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
   * std::size_t size = <a href="afwtraitsrows">rows</a>,
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
   * template&lt;std::size_t size = <a href="afwtraitsrows">rows</a>,
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
   * square matrix of size <a href="afwtraitsrows">rows</a>.<br/></td></tr>
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
   * requires (sizeof...(Args) == <a href="afwtraitsrows">rows</a> * <a href="afwtraitscolumns">columns</a>)
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
  struct MatrixTraits<T> : MatrixTraits<std::decay_t<T>> {};
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
   * <td id="afwtraitsDdimension" class="memItemRight" valign="bottom"><b>dimensions</b></td></tr>
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
   * (row_extent_v&lt;Mean&gt; == row_extent_v&lt;Cov&gt;)</td></tr>
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

} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_INTERFACE_TRAITS_HPP
