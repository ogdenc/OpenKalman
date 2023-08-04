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
 * \brief Forward declaration of LibraryRoutines, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_LIBRARYROUTINES_HPP
#define OPENKALMAN_LIBRARYROUTINES_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::interface
{
  /**
   * \brief An interface to various routines from the library associated with T.
   * \tparam T
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct LibraryRoutines
  {
    /**
     * \brief A Base object within the library of T, if any, from which user-defined objects may be derived (optional).
     */
    template<typename Derived>
    using LibraryBase = std::monostate;


    /**
     * \brief Converts a matrix/array convertible to type <code>T</code> into a dense, writable matrix/array.
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar, std::convertible_to<const std::remove_reference_t<T>&> Arg>
#else
    template<typename Scalar, typename Arg, std::enable_if_t<scalar_type<Scalar> and
      std::is_convertible_v<Arg, const std::remove_reference_t<T>&>, int> = 0>
#endif
    static decltype(auto) convert(Arg&& arg) = delete;


    /**
     * \brief Makes a default, potentially uninitialized, dense, writable matrix or array
     * \details Takes a list of \ref index_descriptor items that specify the size of the resulting object
     * \tparam Ds A list of \ref index_descriptor items
     * \return A default, potentially unitialized, dense, writable matrix or array. Whether the resulting object
     * is a matrix or array may depend on whether T is a matrix or array.
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar, index_descriptor...Ds>
#else
    template<typename Scalar, typename...Ds, std::enable_if_t<scalar_type<Scalar> and (... and index_descriptor<Ds>), int> = 0>
#endif
    static auto make_default(Ds&&...ds) = delete;


    /**
     * \brief Converts Arg (if it is not already) to a native matrix operable within the library associated with T.
     * \details The result should be in a form for which basic matrix operations can be performed within the library for T.
     * If possible, properties such as \ref diagonal_matrix, \ref triangular_matrix, \ref hermitian_matrix,
     * \ref constant_matrix, and \ref constant_diagonal_matrix should be preserved in the resulting object.
     */
    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg) = delete;


    /**
     * \brief Make a writable matrix given a list of elements in row-major order.
     * \tparam Scalar Scalar type of the resulting object.
     * \param d_tup A tuple of index descriptors defining the dimensions of the result.
     * \param args A set of scalar values representing the elements.
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar, index_descriptor...Ds, std::convertible_to<Scalar> ... Args>
#else
    template<typename Scalar, typename...Ds, typename...Args, std::enable_if_t<scalar_type<Scalar> and
      (index_descriptor<Ds> and ...) and std::conjunction_v<std::is_convertible<Args, Scalar>...>, int> = 0>
#endif
    static auto make_from_elements(const std::tuple<Ds...>& d_tup, const Args ... args) = delete;


    /**
     * \brief Create a \ref constant_matrix corresponding to the shape of T (optional).
     * \details Takes a list of \ref index_descriptor items that specify the size of the resulting object
     * \tparam C A \ref scalar_constant (the constant known either at compile time or runtime)
     * \tparam D A list of \ref index_descriptor items
     * \note If this is not defined, it will return an object of type ConstantAdapter.
     */
#ifdef __cpp_concepts
    template<scalar_constant C, index_descriptor...D> requires (sizeof...(D) == IndexibleObjectTraits<T>::max_indices)
#else
    template<typename Scalar, typename C, typename...D, std::enable_if_t<scalar_constant<C> and (index_descriptor<D> and ...) and
      sizeof...(D) == IndexibleObjectTraits<T>::max_indices, int> = 0>
#endif
    static constexpr /*constant_matrix*/ auto
    make_constant_matrix(const C& c, const D&...d) = delete;


    /**
     * \brief Create an \ref identity_matrix.
     * \tparam Scalar The scalar type of the new object
     * \tparam D An \ref index_descriptor defining the size
     * \note If this is not defined, it will return a DiagonalMatrix adapter with a constant diagonal of 1.
     */
#ifdef __cpp_concepts
    template<typename Scalar, index_descriptor D>
#else
    template<typename D, std::enable_if_t<index_descriptor<D>, int> = 0>
#endif
    static constexpr auto make_identity_matrix(D&& d) = delete;


    /**
     * \brief Get a block from a matrix or tensor.
     * \tparam Begin \ref index_value
     * \tparam Size \ref index_value
     * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref index_value.
     * \param size A tuple corresponding to each of indices, each element specifying the dimensions of the extracted block.
     */
#ifdef __cpp_concepts
    template<typename Arg, typename...Begin, typename...Size> requires
      (interface::IndexibleObjectTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin)) and
      (interface::IndexibleObjectTraits<std::decay_t<Arg>>::max_indices == sizeof...(Size))
#else
    template<typename Arg, typename...Begin, typename...Size, std::enable_if_t<
      (interface::IndexibleObjectTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin)) and
      (interface::IndexibleObjectTraits<std::decay_t<Arg>>::max_indices == sizeof...(Size)), int> = 0>
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
      (interface::IndexibleObjectTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin))
#else
    template<typename Arg, typename Block, typename...Begin, typename...Size, std::enable_if_t<
      std::is_convertible_v<Arg&, std::decay_t<T>&> and
      (interface::IndexibleObjectTraits<std::decay_t<Arg>>::max_indices == sizeof...(Begin)), int> = 0>
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


    /**
     * \brief Convert Arg to a set of coordinates in Euclidean space, based on \ref index_descriptor C.
     */
#ifdef __cpp_concepts
    template<std::convertible_to<const std::remove_reference_t<T>&> Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<std::is_convertible_v<Arg, const std::remove_reference_t<T>&> and
      index_descriptor<C>, int> = 0>
#endif
    static constexpr decltype(auto)
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
    static constexpr decltype(auto)
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
    static constexpr decltype(auto)
    wrap_angles(Arg&& arg, const C& c) = delete;
    /* This should be the default:
    {
      return OpenKalman::from_euclidean(OpenKalman::to_euclidean(std::forward<Arg>(arg), c), c);
    }*/


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


#endif //OPENKALMAN_LIBRARYROUTINES_HPP
