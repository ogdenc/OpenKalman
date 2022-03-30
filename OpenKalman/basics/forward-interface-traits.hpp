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


/**
 * \brief The root namespace for OpenKalman interface types.
 */
namespace OpenKalman::interface
{

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
  struct ScalarTypeOf
  {
    /**
     * \typedef type
     * \brief The scalar type of T.
     * \details Example:
     * \code
     * using type = double;
     * \endcode
     */
  };


  /**
   * \internal
   * \brief An interface to the storage array traits of a vector, matrix, matrix expression, or other tensor.
   * \tparam T The tensor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct StorageArrayTraits
  {
    /**
     * \brief The maximum number of indices by which the elements of T are accessible.
     */
    static constexpr std::size_t max_indices = 0;
  };


  /**
   * \internal
   * \brief An interface to the indices of a matrix, or array, expression, or tensor.
   * \tparam T The matrix, array, expression, or tensor.
   * \tparam N The index number
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N>
#else
  template<typename T, std::size_t N, typename = void>
#endif
  struct IndexTraits
  {
    /**
     * \brief The dimension of index N of T, evaluated at compile time.
     * \details For example, a 2-by-3 matrix has dimension 2 in index 0 (rows) and dimension 3 in index 1 (columns).
     */
     static constexpr std::size_t dimension = 0;


    /**
     * \brief Returns the dimension of index N of the argument, evaluated at runtime.
     * \details For example, a 2-by-3 matrix has dimension 2 in index 0 (rows) and dimension 3 in index 1 (columns).
     * \tparam Arg An argument matrix of type T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg,
      const std::add_lvalue_reference_t<std::decay_t<T>>>, int> = 0>
#endif
    static constexpr std::size_t dimension_at_runtime(const Arg& arg) = delete;
  };


  /**
   * \internal
   * \brief An interface to the coordinate system(s) associated with an index of a matrix, array, or expression.
   * \note Any Euclidean types (i.e., regular matrices or other true tensors that operate on a vector space) can rely
   *  on this default definition.
   * \tparam T The matrix, array, expression, or tensor.
   * \tparam N The index number
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N>
#else
  template<typename T, std::size_t N, typename = void>
#endif
  struct CoordinateSystemTraits
  {
    /**
     * \brief The coordinate system type(s) associated with index N of T, evaulated at compile time.
     * \details If the coordinate system is unknown at compile time, use DynamicCoefficients.
     */
     using coordinate_system_types = Axes<IndexTraits<T, N>::dimension>;


    /**
     * \brief Returns the coordinate system type(s) of index N of the argument, evaluated at runtime.
     * \tparam Arg An argument matrix of type T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg,
      const std::add_lvalue_reference_t<std::decay_t<T>>>, int> = 0>
#endif
    static constexpr auto coordinate_system_types_at_runtime(Arg&& arg) = delete;
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
    /// Get element at indices (i...) of matrix arg
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
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
    /// Set element at indices (i...) of matrix arg to s.
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static void set(Arg& arg, const typename ScalarTypeOf<std::decay_t<Arg>>::type& s, I...i) = delete;
  };


  /**
   * \internal
   * \brief Interface to a dense, writable, self-contained matrix or array that is equivalent to T.
   * \details The resulting type is equivalent to T, but may be have a specified shape or scalar type. The interface
   * can set the size or scalar type of the resulting dense matrix based on the parameters (or if they are dynamic,
   * rows and columns can be set to \ref dynamic_size).
   * \tparam T Type upon which the dense matrix will be constructed
   * \tparam rows The specified row dimension of the matrix (defaults to that of T)
   * \tparam columns The specified column dimension of the matrix (defaults to that of T)
   * \tparam Scalar The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T,
    std::size_t rows = IndexTraits<std::decay_t<T>, 0>::dimension,
    std::size_t columns = IndexTraits<std::decay_t<T>, 1>::dimension,
    typename Scalar = typename ScalarTypeOf<std::decay_t<T>>::type>
#else
  template<typename T,
    std::size_t rows = IndexTraits<std::decay_t<T>, 0>::dimension,
    std::size_t columns = IndexTraits<std::decay_t<T>, 1>::dimension,
    typename Scalar = typename ScalarTypeOf<std::decay_t<T>>::type,
    typename = void>
#endif
  struct EquivalentDenseWritableMatrix
  {

    /**
     * \brief Converts a matrix/array convertible to type <code>T</code> into a dense, writable matrix/array of type
     * <code>type</code>.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) convert(Arg&& arg) = delete;


    /**
     * \brief Makes a default, potentially uninitialized, dense, writable matrix or array
     * \details Takes a list of \ref index_descriptor items that specify the size of the resulting object
     * \tparam D A list of \ref index_descriptor items
     * \return A default, potentially unitialized, dense, writable matrix or array. Whether the resulting object
     * is a matrix or array may depend on whether T is a matrix or array.
     */
#ifdef __cpp_concepts
    template<index_descriptor...D>
#else
    template<typename...D, std::enable_if_t<(index_descriptor<D> and ...), int> = 0>
#endif
    static auto make_default(D&&...d) = delete;


    /**
     * \brief Converts Arg (if it is not already) to a native matrix operable within the library associated with T.
     * \details The result must be in a form for which basic matrix operations can be performed within the library for T.
     */
    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg) = delete;

  };


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
#ifdef __cpp_concepts
    template<std::size_t i, std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<std::size_t i, typename Arg, std::enable_if_t<
      std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
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
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
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
   * \internal
   * \brief Interface to a constant scalar matrix in which all elements are zero.
   * \details The resulting type is equivalent to T, but may be have a specified shape or scalar type. The interface
   * can set the size or scalar type of the resulting dense matrix based on the parameters (or if they are dynamic,
   * rows and columns can be set to \ref dynamic_size).
   * \note This definition is optional, and has default behavior if not defined.
   * \tparam T Type upon which the zero matrix will be constructed
   * \tparam rows The specified row dimension of the matrix (defaults to that of T)
   * \tparam columns The specified column dimension of the matrix (defaults to that of T)
   * \tparam Scalar The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T,
    std::size_t rows = IndexTraits<std::decay_t<T>, 0>::dimension,
    std::size_t columns = IndexTraits<std::decay_t<T>, 1>::dimension,
    typename Scalar = typename ScalarTypeOf<std::decay_t<T>>::type>
#else
  template<typename T,
    std::size_t rows = IndexTraits<std::decay_t<T>, 0>::dimension,
    std::size_t columns = IndexTraits<std::decay_t<T>, 1>::dimension,
    typename Scalar = typename ScalarTypeOf<std::decay_t<T>>::type,
    typename = void>
#endif
  struct SingleConstantMatrixTraits
  {
    /**
     * \brief Create a \ref zero_matrix corresponding to the shape of T.
     * \details If T is a \ref dynamic_matrix, you must include one or two runtime dimensions, depending on the number
     * of indices for which dimensions are not specified at compile time. For example, if the row dimension is known at
     * compile time but the column dimension is not, you must specify a single dimension reflecting the number of
     * runtime rows.
     * \note If this is not defined, it will return an object of type ZeroMatrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t>...runtime_dimensions> requires
      (sizeof...(runtime_dimensions) == (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0))
#else
    template<typename...runtime_dimensions, std::enable_if_t<sizeof...(runtime_dimensions) ==
      (rows == dynamic_size ? 1 : 0) + (columns == dynamic_size ? 1 : 0) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int> = 0>
#endif
    static auto make_zero_matrix(runtime_dimensions...e); //< Defined elsewhere


    /**
     * \brief Create a \ref constant_matrix corresponding to the shape of T.
     * \details Takes a list of \ref index_descriptor items that specify the size of the resulting object
     * \tparam D A list of \ref index_descriptor items
     * \note If this is not defined, it will return an object of type ConstantMatrix.
     */
#ifdef __cpp_concepts
    template<auto constant, index_descriptor...D> requires (sizeof...(D) == StorageArrayTraits<T>::max_indices)
#else
    template<auto constant, typename...D, std::enable_if_t<(index_descriptor<D> and ...) and
      sizeof...(D) == StorageArrayTraits<T>::max_indices, int> = 0>
#endif
    static auto make_constant_matrix(D&&...d); //< Defined elsewhere
  };


  /**
   * \brief If T is an object in which all components are a single constant, this is an interface to that constant.
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
   * \internal
   * \brief Interface to an identity matrix.
   * \details The resulting type is equivalent to T, but may be have a specified shape or scalar type. The interface
   * can set the size or scalar type of the resulting identity matrix based on the parameters (or the dimension is
   * dynamic, the dimension can be set to \ref dynamic_size).
   * \tparam T Type upon which the identity matrix will be constructed
   * \tparam dimension The specified row and column dimension of the matrix (defaults to that of T)
   * \tparam Scalar The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T,
    std::size_t dimension = IndexTraits<std::decay_t<T>, 0>::dimension == dynamic_size ?
      IndexTraits<std::decay_t<T>, 1>::dimension : IndexTraits<std::decay_t<T>, 0>::dimension,
    typename Scalar = typename ScalarTypeOf<std::decay_t<T>>::type>
#else
  template<typename T,
    std::size_t dimension = IndexTraits<std::decay_t<T>, 0>::dimension == dynamic_size ?
          IndexTraits<std::decay_t<T>, 1>::dimension : IndexTraits<std::decay_t<T>, 0>::dimension,
    typename Scalar = typename ScalarTypeOf<std::decay_t<T>>::type,
    typename = void>
#endif
  struct SingleConstantDiagonalMatrixTraits
  {
    /**
     * \brief Create an \ref identity_matrix corresponding to the shape of T.
     * \details If T is a \ref dynamic_matrix, you must include a single dimension as an argument,
     * reflecting both the rows and columns of the identity matrix.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::size_t>...runtime_dimensions> requires
      (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0))
#else
    template<typename...runtime_dimensions, std::enable_if_t<(std::is_convertible_v<runtime_dimensions, std::size_t> and ...) and
      (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0)), int> = 0>
#endif
    static auto make_identity_matrix(runtime_dimensions...e) = delete;
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
   * \brief An interface to properties of a diagonal matrix.
   * \details If T is a diagonal matrix, the interface must define static constexpr bool member <code>value</code> as
   * true.
   * \note This class need only be defined for diagonal matrices.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct DiagonalTraits
  {
    /**
     * \brief Whether T is diagonal.
     */
    static constexpr bool is_diagonal = false;
  };


  /**
   * \brief An interface to properties of a triangular matrix.
   * \note This class need only be defined for triangular matrices.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct TriangularTraits
  {
    /**
     * \brief The triangle type of T.
     * \details This trait should propagate from any nested matrices or matrices involved in any expression arguments.
     */
    static constexpr TriangleType triangle_type = TriangleType::none;

    /**
     * \brief Whether T is a triangular adapter.
     */
    static constexpr bool is_triangular_adapter = false;
  };


  /**
   * \brief An interface to properties of a hermitian matrix.
   * \note This class need only be defined for hermitian matrices.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct HermitianTraits
  {
    /**
     * \brief Whether T is hermitian.
     */
    static constexpr bool is_hermitian = false;

    /**
     * \brief The storage type of T, if T is a hermitian adapter.
     * \details This trait should propagate from the nested matrices in any expression or wrapper class. If T is not
     * hermitian, this can be defined as TriangleType::none. T can be hermitian without being a hermitian adapter.
     * But if this value is other than TriangleType::none, then <code>value</code> should be <code>true</code>.
     */
     static constexpr TriangleType adapter_type = TriangleType::none;
  };


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct Subsets
  {
    /**
     * \brief Extract one column from a matrix or other tensor.
     * \details The index of the column may be specified at either compile time <em>or</em> at runtime, but not both.
     * \tparam compile_time_index The index of the column, if specified at compile time
     * \tparam Arg The matrix or other tensor from which the column is to be extracted
     * \tparam runtime_index_t The type of the index of the column, if the index is specified at runtime. This type
     * should be convertible to <code>std::size_t</code>
     * \return A \ref column_vector
     */
    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto) column(Arg&& arg, runtime_index_t...i) = delete;


    /**
     * \brief Extract one row from a matrix or other tensor.
     * \details The index of the row may be specified at either compile time <em>or</em> at runtime, but not both.
     * \tparam compile_time_index The index of the row, if specified at compile time
     * \tparam Arg The matrix or other tensor from which the row is to be extracted
     * \tparam runtime_index_t The type of the index of the row, if the index is specified at runtime. This type
     * should be convertible to <code>std::size_t</code>
     * \return A \ref row_vector
     */
    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto) row(Arg&& arg, runtime_index_t...i) = delete;
  };


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ElementAccess
  {
  };


  /**
   * \brief An interface to necessary array or element-wise operations on matrix T.
   * \tparam T Type of the result matrix, array, or other tensor
   */
  template<typename T>
  struct ArrayOperations
  {
    /**
     * \brief Perform an n-ary array operation on a set of n arguments, possibly with broadcasting.
     * \details If any of the arguments has a lesser order than T, the function must replicate the argument to fill
     * out the full size and shape of T, as necessary, before performing the operation. For example, if T is a 2-by-2
     * matrix and the sole argument is a 2-by-1 column vector, the function must replicate the argument in the
     * horizontal direction to form a 2-by-2 matrix before performing the operation.
     * \tparam sizes The dimension types of T
     * \tparam Operation The n-ary operation taking n arguments, each argument having the same dimensions
     * \tparam Args A set of n arguments
     * \return An object the same size and shape as as T
     */
#ifdef __cpp_concepts
    template<index_descriptor...dims, typename Operation, typename...Args> requires
      (sizeof...(dims) == StorageArrayTraits<std::decay_t<T>>::max_indices)
    static constexpr decltype(auto) n_ary_operation_with_broadcasting(
      const std::tuple<dims...>& d, Operation&&, Args&&...) = delete;
#else
    template<typename...dims, typename Operation, typename...Args, std::enable_if_t<
      (index_descriptor<dims> and ...) and sizeof...(dims) == StorageArrayTraits<std::decay_t<T>>::max_indices, int> = 0>
    static constexpr decltype(auto) n_ary_operation_with_broadcasting(
      const std::tuple<dims...>& d, Operation&&, Args&&...) = delete;
#endif


    /**
     * \brief Use a binary function to reduce a tensor across one or more of its indices.
     * \tparam indices The indices to be reduced. There will be at least one index.
     * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>
     * \tparam Arg The tensor
     * (e.g. std::plus, std::multiplies)
     */
    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto) reduce(BinaryFunction&&, Arg&&) = delete;


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

    /**
     * \brief Convert a column vector into a diagonal matrix.
     * \details An interface need not deal with the following situations, which are already handled by the
     * global \ref OpenKalman::to_diagonal "to_diagonal" function:
     * - a one-by-one matrix
     * - a zero matrix that is known to be square at compile time
     * The interface function <em>should</em> deal with a zero matrix of uncertain size. If the native matrix library
     * does not have a diagonal matrix type, the interface may construct a diagonal matrix using DiagonalMatrix.
     * \tparam Arg A column vector.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg> requires
      (IndexTraits<std::decay_t<T>, 1>::dimension == 1) or (IndexTraits<std::decay_t<T>, 1>::dimension == dynamic_size)
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&> and
      ((IndexTraits<std::decay_t<T>, 1>::dimension == 1) or
       (IndexTraits<std::decay_t<T>, 1>::dimension == dynamic_size)), int> = 0>
#endif
    static constexpr auto
    to_diagonal(Arg&& arg) = delete;


    /**
     * \brief Extract a column vector comprising the diagonal elements of a square matrix.
     * \details An interface need not deal with the following situations, which are already handled by the
     * global \ref OpenKalman::diagonal_of "diagonal_of" function:
     * - an identity matrix
     * - a zero matrix
     * - a constant matrix or constant-diagonal matrix
     * \tparam Arg A square matrix.
     * \returns A column vector
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg> requires
      (IndexTraits<std::decay_t<T>, 0>::dimension == dynamic_size) or
      (IndexTraits<std::decay_t<T>, 1>::dimension == dynamic_size) or
      (IndexTraits<std::decay_t<T>, 0>::dimension == IndexTraits<std::decay_t<T>, 1>::dimension)
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&> and
      ((IndexTraits<std::decay_t<T>, 0>::dimension == dynamic_size) or
       (IndexTraits<std::decay_t<T>, 1>::dimension == dynamic_size) or
       (IndexTraits<std::decay_t<T>, 0>::dimension == IndexTraits<std::decay_t<T>, 1>::dimension)), int> = 0>
#endif
    static constexpr auto
    diagonal_of(Arg&& arg) = delete;

  };


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ModularTransformationTraits
  {
    template<typename...FixedCoefficients, typename Arg, typename...DynamicCoefficients>
    constexpr decltype(auto)
    to_euclidean(Arg&& arg) = delete;

    template<typename...FixedCoefficients, typename Arg, typename...DynamicCoefficients>
    constexpr decltype(auto)
    from_euclidean(Arg&& arg) = delete;

    template<typename...FixedCoefficients, typename Arg, typename...DynamicCoefficients>
    constexpr decltype(auto)
    wrap_angles(Arg&& arg) = delete;
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
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto conjugate(Arg&&) = delete;


    /**
     * \brief Take the transpose of T
     * \tparam Arg An object of type T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto transpose(Arg&&) = delete;


    /**
     * \brief Take the adjoint of T
     * \tparam Arg An object of type T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto adjoint(Arg&&) = delete;


    /**
     * \brief Take the determinant of T
     * \tparam Arg An object of type T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto determinant(Arg&&) = delete;


    /**
     * \brief Take the trace of T
     * \tparam Arg An object of type T
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
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
#ifdef __cpp_concepts
    template<TriangleType t, std::convertible_to<const std::remove_reference_t<T>&> A, typename U, typename Alpha>
#else
    template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
      std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
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
#ifdef __cpp_concepts
    template<TriangleType t, std::convertible_to<const std::remove_reference_t<T>&> A, typename U, typename Alpha>
#else
    template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
      std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) rank_update_triangular(A&&, U&&, const Alpha) = delete;


    /**
     * \brief Solve the equation AX = B for X, which may or may not be a unique solution.
     * \tparam must_be_unique Determines whether the function throws an exception if the solution X is non-unique
     * (e.g., if the equation is under-determined)
     * \tparam must_be_exact Determines whether the function throws an exception if it cannot return an exact solution,
     * such as if the equation is over-determined. * If <code>false<code>, then the function will return an estimate
     * instead of throwing an exception.
     * \tparam A The matrix A in the equation AX = B
     * \tparam B The matrix B in the equation AX = B
     * \return The solution X of the equation AX = B. If <code>must_be_unique</code>, then the function can return
     * any valid solution for X.
     */
#ifdef __cpp_concepts
    template<bool must_be_unique = false, bool must_be_exact = false,
      std::convertible_to<const std::remove_reference_t<T>&> A, typename B>
#else
    template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B, std::enable_if_t<
      std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) solve(A&&, B&&) = delete;

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
   * \details This class includes key information about a matrix or matrix expression, such as its dimension,
   * coefficient types, etc.
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
   * \details This class includes key information about a matrix or matrix expression, such as its dimension,
   * coefficient types, etc.
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
