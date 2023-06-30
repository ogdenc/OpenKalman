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


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to traits of a particular index of a matrix, or array, expression, or tensor.
   * \tparam T The matrix, array, expression, or tensor.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct IndexTraits
  {
    /**
     * \brief The maximum number of indices by which the elements of T are accessible.
     * \details T may optionally be accessible by fewer indices.
     */
    static constexpr std::size_t max_indices = 0;


    /**
     * \brief Get an \ref index_descriptor for index N of an argument.
     * \tparam N The index
     * \param arg An indexible object (tensor, matrix, vector, etc.)
     * \return an \ref index_descriptor (either fixed or dynamic)
     */
#ifdef __cpp_concepts
    template<std::size_t N, std::convertible_to<const std::remove_reference_t<T>&> Arg>
    static constexpr index_descriptor auto get_index_descriptor(const Arg& arg)
#else
    template<std::size_t N, typename Arg, std::enable_if_t<std::is_convertible_v<Arg,
      const std::add_lvalue_reference_t<std::decay_t<T>>>, int> = 0>
    static constexpr auto get_index_descriptor(const Arg& arg)
#endif
    {
      return Dimensions<0>{};
    }


    /**
     * \brief Whether T is one-by-one (optional).
     * \details This can be useful because some dynamic matrix types erase the shape info about their nested matrix,
     * such that it does not preserve the knowledge that the object cannot be one-by-one.
     */
    template<Likelihood b>
    static constexpr bool is_one_by_one = false;


    /**
     * \brief Whether all dimensions of T are the same and type-equivalent (optional).
     * \details This can be useful because some dynamic matrix types erase the shape info about their nested matrix,
     * such that it does not preserve the knowledge that the object cannot be square.
     */
    template<Likelihood b>
    static constexpr bool is_square = false;

  };


  /**
   * \brief An interface to features of individual elements of indexible object T using indices I... of type std::size_t.
   * \detail The interface may define static member function <code>get</code>  and <code>set</code> with one or two indices. If
   * getting or setting an element is not possible, leave <code>get</code> or <code>set</code> undefined, respectively.
   * \tparam I The indices (each of type std::size_t)
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct Elements
  {
    /**
     * \typedef scalar_type
     * \brief The scalar type of T (e.g., double, int).
     * \details Example:
     * \code
     * using scalar_type = double;
     * \endcode
     */



    /**
     * \brief Get element at indices (i...) of matrix arg. This should preferably return a non-const lvalue reference, if possible.
     * \details This function, or the library, is responsible for any bounds checking.
     * \returns an element or reference to an element
     *
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::decay_t<T>&> Arg, std::convertible_to<const std::size_t>...I>
#else
    template<typename Arg, typename...I, std::enable_if_t<
      std::is_convertible_v<Arg, const std::decay_t<T>&> and (std::is_convertible_v<I, const std::size_t> and ...), int> = 0>
#endif
    static constexpr decltype(auto) get(Arg&& arg, I...i) = delete;


    /**
     * \brief Set element at indices (i...) of matrix arg to s.
     * \details This function, or the library, is responsible for any bounds checking.
     *
     */
#ifdef __cpp_concepts
    template<std::convertible_to<std::decay_t<T>> Arg, std::convertible_to<const std::size_t>...I>
      requires (not std::is_const_v<std::remove_reference_t<Arg>>)
#else
    template<typename Arg, typename...I, std::enable_if_t<
      std::is_convertible_v<Arg, std::decay_t<T>> and (std::is_convertible_v<I, const std::size_t> and ...) and
      not std::is_const_v<std::remove_reference_t<Arg>>, int> = 0>
#endif
    static void set(Arg& arg, const typename Elements<std::decay_t<Arg>>::scalar_type& s, I...i) = delete;
  };


  /**
   * \internal
   * \brief Interface to a dense, writable, self-contained matrix or array that is equivalent to T.
   * \details The resulting type is equivalent to T, but may be have a specified shape or scalar type. The interface
   * can set the size or scalar type of the resulting dense matrix based on the parameters (or if they are dynamic,
   * rows and columns can be set to \ref dynamic_size).
   * \tparam T Type upon which the dense matrix will be constructed
   * \tparam Scalar The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T, typename Scalar = typename Elements<std::decay_t<T>>::scalar_type>
#else
  template<typename T, typename Scalar = typename Elements<std::decay_t<T>>::scalar_type, typename = void>
#endif
  struct EquivalentDenseWritableMatrix
  {
    /**
     * \brief Whether T is already a writable, self-contained matrix or array.
     */
    static constexpr bool is_writable = false;


    /**
     * \brief Converts a matrix/array convertible to type <code>T</code> into a dense, writable matrix/array.
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
    template<typename...D>
    static auto make_default(D&&...d) = delete;


    /**
     * \brief Converts Arg (if it is not already) to a native matrix operable within the library associated with T.
     * \details The result should be in a form for which basic matrix operations can be performed within the library for T.
     * If possible, properties such as \ref diagonal_matrix, \ref triangular_matrix, \ref hermitian_matrix,
     * \ref constant_matrix, and \ref constant_diagonal_matrix should be preserved in the resulting object.
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
     * \details An example of an internal runtime parameter might be indices for start locations, or sizes, for an
     * expression representing a block or sub-matrix within a matrix. If unknown, the value of <code>true</code> is
     * the safest and will prevent unintended dangling references.
     * \note If this is not defined, T will be treated as if it is defined and true. This parameter can ignore whether
     * any nested matrices, themselves, have internal runtime parameters.
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
   * \tparam Scalar The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T, typename Scalar = typename Elements<std::decay_t<T>>::scalar_type>
#else
  template<typename T, typename Scalar = typename Elements<std::decay_t<T>>::scalar_type, typename = void>
#endif
  struct SingleConstantMatrixTraits
  {
    /**
     * \internal
     * \brief Create a \ref constant_matrix corresponding to the shape of T (optional).
     * \details Takes a list of \ref index_descriptor items that specify the size of the resulting object
     * \tparam C A \ref scalar_constant (the constant known either at compile time or runtime)
     * \tparam D A list of \ref index_descriptor items
     * \note If this is not defined, it will return an object of type ConstantAdapter.
     */
#ifdef __cpp_concepts
    template<scalar_constant C, index_descriptor...D> requires (sizeof...(D) == IndexTraits<T>::max_indices)
#else
    template<typename C, typename...D, std::enable_if_t<scalar_constant<C> and (index_descriptor<D> and ...) and
      sizeof...(D) == IndexTraits<T>::max_indices, int> = 0>
#endif
    static constexpr /*constant_matrix*/ auto
    make_constant_matrix(const C& c, const D&...d) = delete;
  };


  /**
   * \brief Traits for \ref constant_matrix or \ref constant_diagonal_matrix objects.
   * \details If T's constant is known only at runtime, this class must be constructible from an object of type T.
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
     * \brief Constructor for the case that T is only be derivable at runtime.
     */
    SingleConstant(const std::decay_t<T>&) = delete;

    /**
     * \brief If T has a constant value across all elements, return that constant.
     * \details The return type must be convertible to scalar_type_of_t<T>. If T's elements are a constant known
     * only at runtime, this should be a non-static function.
     */
    constexpr auto get_constant(const T&) = delete;

    /**
     * \brief If T has a constant diagonal value, return that constant.
     * \details The return type must be convertible to scalar_type_of_t<T>. If T's elements are a constant known
     * only at runtime, this should be a non-static function.
     */
    constexpr auto get_constant_diagonal(const T&) = delete;

  };


  /// \brief Deduction guide for \ref interface::SingleConstant
  template<typename T>
  explicit SingleConstant(const T&) -> SingleConstant<T>;


  /**
   * \internal
   * \brief Interface to an identity matrix.
   * \details The resulting type is equivalent to T, but may be have a specified shape or scalar type. The interface
   * can set the size or scalar type of the resulting identity matrix based on the parameters (or the dimension is
   * dynamic, the dimension can be set to \ref dynamic_size).
   * \tparam T Type upon which the identity matrix will be constructed
   * \tparam Scalar The specified scalar type of the matrix (defaults to that of T)
   */
#ifdef __cpp_concepts
  template<typename T, typename Scalar = typename Elements<std::decay_t<T>>::scalar_type>
#else
  template<typename T, typename Scalar = typename Elements<std::decay_t<T>>::scalar_type, typename = void>
#endif
  struct SingleConstantDiagonalMatrixTraits
  {
    /**
     * \brief Create an \ref identity_matrix.
     * \tparam D An \ref index_descriptor defining the size
     * \note If this is not defined, it will return a DiagonalMatrix adapter with a constant diagonal of 1.
     */
/*#ifdef __cpp_concepts
    template<index_descriptor D>
#else
    template<typename D, std::enable_if_t<index_descriptor<D>, int> = 0>
#endif
    static constexpr auto make_identity_matrix(D&& d); //< Defined elsewhere*/
  };


  /**
   * \brief An interface to properties of a triangular  or diagonal matrix.
   * \note This class need only be defined for triangular or diagonal matrices.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct TriangularTraits
  {
    /**
     * \brief Whether T is triangular or diagonal, having a triangle type of t.
     * \details This trait should propagate from any nested matrices or matrices involved in any expression arguments.
     * \tparam t The \ref TriangleType
     * \tparam b The \ref Likelihood. If <code>b == Likelihood::definitely</code>, then T's triangle type is known at compile time.
     * If <code>b == Likelihood::maybe</code>, then T's triangle type is determined at runtime (for example, T might be
     * triangular if and only iff it is a square matrix, but it is unknown whether T is square).
     */
    template<TriangleType t, Likelihood b>
    static constexpr bool is_triangular = false;

    /**
     * \brief Whether T is a triangular adapter (defaults to false, if omitted).
     * \details This is not a guarantee that the matrix is triangular, because it could be dynamically non-square.
     */
    static constexpr bool is_triangular_adapter = false;

    /**
     * \brief Whether T is a \ref diagonal_adapter (defaults to false, if omitted).
     * \details The likelihood b is available if it is not known whether the nested matrix is a column vector
     */
    template<Likelihood b>
    static constexpr bool is_diagonal_adapter = false;

    /**
     * \brief Create a \ref triangular_matrix from a square matrix.
     * \details This is used by the function OpenKalman::make_triangular_matrix. This can be left undefined if
     * - Arg is already triangular and of a TriangleType compatible with t, or
     * - the intended result is for Arg to be wrapped in an \ref Eigen::TriangularMatrix (which will happen automatically).
     * \tparam t The intended \ref TriangleType of the result.
     * \tparam Arg A square matrix to be made triangular.
     */
#ifdef __cpp_concepts
    template<TriangleType t, std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<TriangleType t, typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto make_triangular_matrix(Arg&& arg) = delete;
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
     * \brief The hermitian-adapter storage type of T, if any (optional).
     * \details This is not a guarantee that the matrix is hermitian, because it could be dynamically non-square.
     * If T is not a hermitian adapter, this should be omitted. Permissible values are HermitianAdapterType::upper and
     * HermitianAdapterType::lower.
     */
     //static constexpr HermitianAdapterType adapter_type = HermitianAdapterType::lower;

     /**
      * \brief Make a hermitian adapter.
      */
     template<HermitianAdapterType t, typename Arg>
     static constexpr auto make_hermitian_adpater(Arg&& arg) = delete;
  };


#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct Subsets
  {
    /**
     * \brief Get a block from a matrix or tensor.
     * \tparam Begin \ref index_value
     * \tparam Size \ref index_value
     * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref index_value.
     * \param size A tuple corresponding to each of indices, each element specifying the dimensions of the extracted block.
     */
#ifdef __cpp_concepts
    template<typename Arg, typename...Begin, typename...Size> requires
      (interface::IndexTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin)) and
      (interface::IndexTraits<std::decay_t<Arg>>::max_indices == sizeof...(Size))
#else
    template<typename Arg, typename...Begin, typename...Size, std::enable_if_t<
      (interface::IndexTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin)) and
      (interface::IndexTraits<std::decay_t<Arg>>::max_indices == sizeof...(Size)), int> = 0>
#endif
    static decltype(auto) get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size) = delete;


    /**
     * \brief Set a block from a \ref writable matrix or tensor.
     * \tparam Arg The matrix or tensor to be modified.
     * \tparam Block A block to be copied into Arg at a particular location.
     * \tparam Begin \ref index_value corresponding to each of indices, specifying the beginning \ref index_value.
     * \returns An lvalue reference to Arg.
     */
#ifdef __cpp_concepts
    template<typename Arg, typename Block, typename...Begin> requires std::convertible_to<Arg&, std::decay_t<T>&> and
      (interface::IndexTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin))
#else
    template<typename Arg, typename Block, typename...Begin, typename...Size, std::enable_if_t<
      std::is_convertible_v<Arg&, std::decay_t<T>&> and
      (interface::IndexTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin)), int> = 0>
#endif
    static Arg& set_block(Arg& arg, Block&& block, Begin...begin) = delete;


    /**
     * \brief Set only a triangular (or diagonal) portion taken from another matrix to a \ref writable matrix.
     * \note This is optional.
     * \tparam t The TriangleType (upper, lower, or diagonal)
     * \tparam A The matrix or tensor to be set
     * \tparam B A matrix or tensor to be copied from, which may or may not be triangular
     */
#ifdef __cpp_concepts
    template<TriangleType t, typename A, typename B> requires std::convertible_to<A&&, std::decay_t<T>&>
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<std::is_convertible_v<A&&, std::decay_t<T>&>>>
#endif
    static decltype(auto) set_triangle(A&& a, B&& b) = delete;
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
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr decltype(auto)
    to_diagonal(Arg&& arg) = delete;
    /* This should be the default:
    {
      return DiagonalMatrix<passable_t<Arg>> {std::forward<Arg>(arg)};
    }*/


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
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr decltype(auto)
    diagonal_of(Arg&& arg) = delete;

  };


  /**
   * \brief An interface to necessary array or element-wise operations on matrix T.
   * \tparam T Type of the result matrix, array, or other tensor
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ArrayOperations
  {
    /**
     * \brief Perform an n-ary array operation on a set of n arguments, possibly with broadcasting.
     * \details The index descriptors d_tup define the size of the resulting matrix. If any of the arguments has a
     * lesser order than indicated by d_tup, the function must replicate the argument to fill
     * out the full size and shape specified by Ds, as necessary, before performing the operation. For example, if
     * d_tup is Dimensions<2> and Dimensions<2> and the sole argument is a 2-by-1 column vector, the function must
     * replicate the argument in the horizontal direction to form a 2-by-2 matrix before performing the operation.
     * \note This is optional and should be left undefined to the extent the native library does not provide this
     * functionality.
     * \param d_tup A tuple of index descriptors (of type Ds) defining the resulting tensor
     * \tparam Operation The n-ary operation taking n arguments, each argument having the same dimensions
     * \tparam Args A set of n arguments
     * \return An object with size and shape defined by d_tup and with elements defined by the operation
     */
    template<typename...Ds, typename Operation, typename...Args>
    static auto n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&&, Args&&...) = delete;


    /**
     * \brief Fill an array or tensor using an n-ary operation that also takes indices as arguments.
     * \details The n-ary operation results in elements that can be index-dependent.
     * \note This is optional and should be left undefined to the extent the native library does not provide this
     * functionality.
     * \param d_tup A tuple of index descriptors (of type Ds) defining the resulting tensor
     * \tparam Operation An n-ary operation taking n arguments as well as the indices defining T
     * \tparam Args A set of n arguments
     * \return An object with size and shape defined by d_tup and with elements defined by the n-ary operation
     */
    template<typename...Ds, typename Operation, typename...Args>
    static auto n_ary_operation_with_indices(const std::tuple<Ds...>& d_tup, Operation&&, Args&&...) = delete;


    /**
     * \brief Use a binary function to reduce a tensor across one or more of its indices.
     * \details The binary function is assumed to be associative, so any order of operation is permissible.
     * \tparam indices The indices to be reduced. There will be at least one index.
     * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>
     * (e.g. std::plus, std::multiplies)
     * \tparam Arg The tensor
     * \returns A vector or tensor with reduced dimensions. If <code>indices...</code> includes every index of Arg
     * (thus calling for a complete reduction), the function may return either a scalar value or a one-by-one matrix.
     */
    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto) reduce(BinaryFunction&&, Arg&&) = delete;
  };


  /**
   * \brief Traits relating to wrapping angles and other modular coordinate types.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct ModularTransformationTraits
  {
    /**
     * \brief Convert Arg to a set of coordinates in Euclidean space, based on \ref index_descriptor C.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&> and
      index_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, const C& c) = delete;
    /* This should be the default:
    {
      return ToEuclideanExpr<C, passable_t<Arg>> {std::forward<Arg>(arg), c};
    }*/


    /**
     * \brief Convert Arg from a set of coordinates in Euclidean space, based on \ref index_descriptor C.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&> and
      index_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, const C& c) = delete;
    /* This should be the default:
    {
      return FromEuclideanExpr<C, passable_t<Arg>> {std::forward<Arg>(arg), c};
    }*/


    /**
     * \brief Wrap Arg based on \ref index_descriptor C.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&> and
      index_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, const C& c) = delete;
    /* This should be the default:
    {
      return OpenKalman::from_euclidean(OpenKalman::to_euclidean(std::forward<Arg>(arg), c), c);
    }*/
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
     * \brief Perform an element-by-element sum of compatible tensors
     * \tparam A A tensor of type T
     * \tparam B Another tensor of the same dimensions as A
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> A, typename B>
#else
    template<typename A, typename B, std::enable_if_t<std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto sum(A&& a, B&& b) = delete;


    /**
     * \brief Perform a contraction involving two compatible tensors
     * \tparam A A tensor of type T
     * \tparam B Another tensor of the same dimensions as A
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> A, typename B>
#else
    template<typename A, typename B, std::enable_if_t<std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto contract(A&& a, B&& b) = delete;


    /**
     * \brief Perform an in-place contraction involving two compatible tensors
     * \tparam A A tensor of type T
     * \tparam B Another tensor of the same dimensions as A
     * \return A reference to A
     */
#ifdef __cpp_concepts
    template<bool on_the_right, std::convertible_to<const std::remove_reference_t<T>&> A, typename B>
#else
    template<bool on_the_right, typename A, typename B, std::enable_if_t<std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr A& contract_in_place(A& a, B&& b) = delete;


    /**
     * \brief Take the Cholesky factor of matrix Arg
     * \tparam triangle_type The \ref TriangleType of the result.
     * \param a A matrix of type T
     * \return A matrix t where tt<sup>T</sup> = a (if triangle_type == TriangleType::lower) or
     * t<sup>T</sup>t = a (if triangle_type == TriangleType::upper).
     */
#ifdef __cpp_concepts
    template<TriangleType triangle_type, std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<TriangleType triangle_type, typename Arg, std::enable_if_t<
      std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto cholesky_factor(Arg&& a) = delete;


    /**
     * \brief Do a rank update on a hermitian matrix.
     * \note This is preferably (but not necessarily) performed as an in-place operation.
     * \details A must be a \ref hermitian_matrix.
     * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
     * - If A is a non-const lvalue reference, it should be updated in place if possible. Otherwise, the function may return a new matrix.
     * \tparam significant_triangle The triangle which is significant (or TriangleType::any if both are significant)
     * \tparam A A writable object (same library as type T) in which triangle t is significant.
     * \tparam U The update vector or matrix.
     * \returns an updated native, writable matrix in hermitian form.
     */
#ifdef __cpp_concepts
    template<HermitianAdapterType significant_triangle, std::convertible_to<const std::remove_reference_t<T>&> A, typename U, typename Alpha>
#else
    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha, std::enable_if_t<
      std::is_convertible_v<A, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) rank_update_self_adjoint(A&&, U&&, const Alpha) = delete;


    /**
     * \brief Do a rank update on a triangular matrix.
     * \note This is preferably (but not necessarily) performed as an in-place operation.
     * \details A must be a triangular matrix.
     * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
     * returning the updated A.
     * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
     * - If A is a non-const lvalue reference, it should be updated in place if possible. Otherwise, the function may return a new matrix.
     * \tparam triangle The triangle (upper or lower)
     * \tparam A An object of type T, which is either triangular or dense-writable.
     * \tparam U The update vector or matrix.
     * \returns an updated native, writable matrix in triangular (or diagonal) form.
     */
#ifdef __cpp_concepts
    template<TriangleType triangle, std::convertible_to<const std::remove_reference_t<T>&> A, typename U, typename Alpha>
#else
    template<TriangleType triangle, typename A, typename U, typename Alpha, std::enable_if_t<
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


    /**
     * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
     * \tparam A The matrix to be decomposed
     * \returns L as a lower \ref triangular_matrix
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto LQ_decomposition(Arg&&) = delete;


    /**
     * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
     * \tparam A The matrix to be decomposed
     * \returns U as an upper \ref triangular_matrix
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static constexpr auto QR_decomposition(Arg&&) = delete;

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
