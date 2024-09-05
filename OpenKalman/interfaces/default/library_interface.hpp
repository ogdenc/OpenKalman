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
 * \brief Forward declaration of library_interface, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_LIBRARY_INTERFACE_HPP
#define OPENKALMAN_LIBRARY_INTERFACE_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<typename LibraryObject>
#else
  template<typename LibraryObject, typename>
#endif
  struct library_interface
  {
#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /**
     * \brief The base class within the library of LibraryObject for custom wrappers (optional).
     * \details This is used when a library requires custom objects to derive from a particular base class.
     * For example, the Eigen library requires objects to derive from classes such as <code>Eigen::EigenBase</code>,
     * <code>Eigen::MatrixBase</code>, or <code>Eigen::ArrayBase</code>.
     * The particular base class within a given library can depend on the type of LibraryObject
     * (e.g., whether LibraryObject is a matrix or array).
     */
    template<typename Derived>
    using LibraryBase = std::monostate;


    /**
     * \brief Get a scalar component of Arg at a given set of indices.
     * \details The indices are are in the form of a ranged object accessible by an iterator.
     * \tparam Indices A ranged object satisfying std::ranges::input_range, which contains exactly <code>count_indices(arg)</code> indices.
     * \returns an element or reference to a component of Arg, as a \ref scalar_constant (preferably as a non-const lvalue reference)
     * \note Mandatory. Also, this function, or the library, is responsible for any optional bounds checking.
     *
     */
#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>>
    static constexpr scalar_constant decltype(auto)
#else
    template<typename Arg, typename Indices>
    static constexpr decltype(auto)
#endif
    get_component(Arg&& arg, const Indices& indices) = delete;


    /**
     * \brief Set a component of Arg at a given set of indices to scalar value s.
     * \details The indices are are in the form of a ranged object accessible by an iterator.
     * \tparam Indices A ranged object satisfing std::ranges::input_range, which contains exactly
     * <code>count_indices(arg)</code> indices.
     * \note Mandatory. Also, this function, or the library, is responsible for any optional bounds checking.
     *
     */
#ifdef __cpp_lib_ranges
    template<indexible Arg, std::ranges::input_range Indices> requires index_value<std::ranges::range_value_t<Indices>>
#else
    template<typename Arg, typename Indices>
#endif
    static void
    set_component(Arg& arg, const scalar_type_of_t<Arg>& s, const Indices& indices) = delete;


    /**
     * \brief Converts Arg (if it is not already) to a native matrix operable within the library associated with LibraryObject.
     * \details The result should be in a form for which basic matrix operations can be performed within the library for LibraryObject.
     * This should be a lightweight transformation that does not require copying of elements.
     * If possible, properties such as \ref diagonal_matrix, \ref triangular_matrix, \ref hermitian_matrix,
     * \ref constant_matrix, and \ref constant_diagonal_matrix should be preserved in the resulting object.
     * \note if not defined, a call to \ref OpenKalman::to_native_matrix will construct a \ref LibraryWrapper.
     */
#ifdef __cpp_concepts
    static decltype(auto)
    to_native_matrix(auto&& arg) = delete;
#else
    template<typename Arg>
    static decltype(auto)
    to_native_matrix(Arg&& arg) = delete;
#endif


    /**
     * \brief Assign (copy or move) the elements of an indexible object to another indexible object.
     * \tparam To The \ref indexible object to be assigned.
     * \tparam From The \ref indexible object from which to assign.
     */
#ifdef __cpp_concepts
    template<writable To, indexible From>
#else
    template<typename To, typename From, std::enable_if_t<indexible<To> and indexible<From>, int> = 0>
#endif
    static void
    assign(M& a, From&& b) = delete;


    /**
     * \brief Makes a default, potentially uninitialized, dense, writable matrix or array
     * \details Takes a list of \ref vector_space_descriptor objects that specify the size of the resulting object
     * \tparam layout the \ref Layout of the result, which may be Layout::left, Layout::right, or
     * Layout::none (which indicates the default layout for the library).
     * \tparam Scalar The scalar value of the result.
     * \param ds A list of \ref vector_space_descriptor items
     * \return A default, potentially uninitialized, dense, writable object.
     * \note The interface may base the return value on any properties of LibraryObject (e.g., whether LibraryObject is a matrix or array).
     */
#ifdef __cpp_concepts
    template<Layout layout, scalar_type Scalar> requires (layout != Layout::stride)
    static auto
    make_default(vector_space_descriptor auto&&...ds) = delete;
#else
    template<Layout layout, typename Scalar, typename...Ds>
    static auto
    make_default(Ds&&...ds) = delete;
#endif


    /**
     * \brief Fill a writable matrix with a list of elements.
     * \tparam Arg The \ref writable object to be filled.
     * \tparam layout The \ref Layout of the listed elements, which may be \ref Layout::left or \ref Layout::right.
     * \param scalars A set of scalar values representing the elements. There must be exactly the right number of elements
     * to fill Arg.
     */
#ifdef __cpp_concepts
    template<Layout layout, writable Arg> requires (layout == Layout::right) or (layout == Layout::left)
    static void
    fill_components(Arg& arg, const std::convertible_to<scalar_type_of_t<Arg>> auto ... scalars) = delete;
#else
    template<Layout layout, typename Arg, typename...Scalars, std::enable_if_t<
      writable<Arg> and (layout == Layout::right) or (layout == Layout::left), int> = 0>
    static void
    fill_components(Arg& arg, const Scalars...scalars) = delete;
#endif


    /**
     * \brief Create a \ref constant_matrix of a given shape (optional).
     * \details Takes a list of \ref vector_space_descriptor items that specify the size of the resulting object
     * \param c A \ref scalar_constant (either static or dynamic)
     * \param d A list of \ref vector_space_descriptor items
     * \note If this is not defined, calls to <code>OpenKalman::make_constant</code> will return an object of type ConstantAdapter.
     */
#ifdef __cpp_concepts
    static constexpr constant_matrix auto
    make_constant(const scalar_constant auto& c, vector_space_descriptor auto&&...d) = delete;
#else
    template<typename C, typename...D>
    static constexpr auto
    make_constant(const C& c, D&&...d) = delete;
#endif


    /**
     * \brief Create a generalized \ref identity_matrix of a given shape(optional).
     * \details This is a generalized identity matrix that need not be square, but every non-diagonal element must be zero.
     * \note If not defined, an identity matrix is a \ref DiagonalMatrix adapter with a constant diagonal of 1.
     * \tparam Scalar The scalar type of the new object
     * \param d A \ref vector_space_descriptor object defining the size
     */
#ifdef __cpp_concepts
    template<scalar_type Scalar>
    static constexpr identity_matrix auto
    make_identity_matrix(vector_space_descriptor auto&&...d) = delete;
#else
    template<typename Scalar, typename...D>
    static constexpr auto
    make_identity_matrix(D&&...d) = delete;
#endif


    /**
     * \brief Create a \ref triangular_matrix from a square matrix.
     * \details This is used by the function OpenKalman::make_triangular_matrix. This can be left undefined if
     * - Arg is already triangular and of a TriangleType compatible with t, or
     * - the intended result is for Arg to be wrapped in an \ref Eigen::TriangularMatrix (which will happen automatically).
     * \tparam t The intended \ref TriangleType of the result.
     * \param arg A square matrix to be wrapped in a triangular adapter.
     */
#ifdef __cpp_concepts
    template<TriangleType t>
    static constexpr auto
    make_triangular_matrix(indexible auto&& arg) = delete;
#else
    template<TriangleType t, typename Arg>
    static constexpr auto
    make_triangular_matrix(Arg&& arg) = delete;
#endif


    /**
     * \brief Make a hermitian adapter.
     * \details This is used by the function OpenKalman::make_hermitian_matrix. This can be left undefined if
     * - Arg is already hermitian and of a HermitianAdapterType compatible with t, or
     * - the intended result is for Arg to be wrapped in an \ref Eigen::SelfAdjointMatrix (which will happen automatically).
     * \tparam t The intended \ref HermitianAdapterType of the result.
     * \param arg A square matrix to be wrapped in a hermitian hermitian.
     */
#ifdef __cpp_concepts
     template<HermitianAdapterType t>
     static constexpr auto
     make_hermitian_adapter(indexible auto&& arg) = delete;
#else
     template<HermitianAdapterType t, typename Arg>
     static constexpr auto
     make_hermitian_adapter(Arg&& arg) = delete;
#endif


    /**
     * \brief Get a block from a matrix or tensor.
     * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref index_value.
     * \param size A tuple corresponding to each of indices, each element specifying the size (as an \ref index_value) of the extracted block.
     */
#ifdef __cpp_concepts
    template<indexible Arg, index_value...Begin, index_value...Size> requires
      (index_count_v<Arg> == sizeof...(Begin)) and (index_count_v<Arg> == sizeof...(Size))
    static indexible decltype(auto)
#else
    template<typename Arg, typename...Begin, typename...Size>
    static decltype(auto)
#endif
    get_slice(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size) = delete;


    /**
     * \brief Set a block from a \ref writable matrix or tensor.
     * \param arg The matrix or tensor to be modified.
     * \param block A block to be copied into Arg at a particular location.
     * Note: This may not necessarily be within the same library as arg, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \param begin \ref index_value corresponding to each of indices, specifying the beginning \ref index_value.
     * \returns An lvalue reference to arg.
     */
#ifdef __cpp_concepts
    template<writable Arg, indexible Block, index_value...Begin> requires
      (index_count_v<Block> == sizeof...(Begin)) and (index_count_v<Arg> == sizeof...(Begin))
#else
    template<typename Arg, typename Block, typename...Begin>
#endif
    static void
    set_slice(Arg& arg, Block&& block, const Begin&...begin) = delete;


    /**
     * \brief Set only a triangular (or diagonal) portion of a \ref writable matrix with elements of another matrix.
     * \details Neither a nor b need to be square matrices.
     * \note This is optional.
     * \tparam t The TriangleType (upper, lower, or diagonal)
     * \param a The matrix or tensor to be set
     * \param b A matrix or tensor to be copied from, which may or may not be triangular.
     * \return a as altered
     */
#ifdef __cpp_concepts
    template<TriangleType t>
    static void
    set_triangle(writable auto& a, indexible auto&& b) = delete;
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<writable<A> and indexible<B>, int> = 0>
    static void
    set_triangle(A& a, B&& b) = delete;
#endif


    /**
     * \brief Convert a column vector (or column slice for rank>2 tensors) into a diagonal matrix (optional).
     * \note If this is not defined, calls to <code>OpenKalman::to_diagonal</code> will construct a \ref DiagonalMatrix.
     * \details An interface need not deal with an object known to be \ref one_dimensional at compile time.
     * \tparam Arg A column vector.
     */
#ifdef __cpp_concepts
    static constexpr diagonal_matrix auto
    to_diagonal(vector<0, Qualification::depends_on_dynamic_shape> auto&& arg) = delete;
#else
    template<typename Arg>
    static constexpr auto
    to_diagonal(Arg&& arg) = delete;
#endif


    /**
     * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
     * \details An interface need not deal with the following situations, which are already handled by the
     * global \ref OpenKalman::diagonal_of "diagonal_of" function:
     * - an identity matrix
     * - a zero matrix
     * - a constant matrix or constant-diagonal matrix
     * \param arg A square matrix.
     * \returns A column vector
     */
#ifdef __cpp_concepts
    static constexpr vector auto
    diagonal_of(square_shaped<Qualification::depends_on_dynamic_shape> auto&& arg) = delete;
#else
    template<typename Arg>
    static constexpr auto
    diagonal_of(Arg&& arg) = delete;
#endif


    /**
     * \brief Broadcast an object by replicating it by factors specified for each index.
     * \details The operation may increase the order of the object by specifying factors beyond the order of the argument.
     * \param arg The object.
     * \param factors A set of factors indicating the increase in size of each index. There must be one factor per
     * index, and there may also be additional factors if the tensor order is to be expanded
     * \todo Connect this with a general function and possibly with n_ary_operation
     */
#ifdef __cpp_concepts
    static indexible auto
    broadcast(indexible auto&& arg, const index_value auto&...factors) = delete;
#else
    template<typename Arg, typename...Factors>
    static auto
    broadcast(Arg&& arg, const Factors&...factors) = delete;
#endif


    /**
     * \brief Perform an n-ary array operation on a set of n arguments.
     * \details The \ref vector_space_descriptor tuple d_tup defines the size of the resulting matrix.
     * \note This is optional and should be left undefined to the extent the native library does not provide this
     * functionality.
     * \param d_tup A tuple of \ref vector_space_descriptor (of type Ds) defining the resulting tensor
     * \tparam Operation The n-ary operation taking n scalar arguments and (optionally) <code>sizeof...(Ds)</code> indices.
     * Examples:
     * - <code>template<scalar_type...X> operation(const X&...)</code>
     * - <code>template<scalar_type...X, index_value...I> operation(const X&..., I...)</code>
     * \param args A set of n indexible arguments, each having the same dimensions.
     * \return An object with size and shape defined by d_tup and with elements defined by the operation
     * \todo Eliminate the Ds?
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Ds, typename Operation, indexible...Args> requires
      std::invocable<Operation&&, scalar_type_of_t<Args>...> or
      std::invocable<Operation&&, scalar_type_of_t<Args>..., std::conditional_t<true, std::size_t, Ds>...>
    static compatible_with_vector_space_descriptors<Ds...> auto
#else
    template<typename...Ds, typename Operation, typename...Args>
    static auto
#endif
    n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& op, Args&&...args) = delete;


    /**
     * \brief Use a binary function to reduce a tensor across one or more of its indices.
     * \details The binary function is assumed to be associative, so any order of operation is permissible.
     * \tparam indices The indices to be reduced. There will be at least one index.
     * \param op A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>
     * (e.g. std::plus, std::multiplies)
     * \param arg An object to be reduced
     * \returns A vector or tensor with reduced dimensions. If <code>indices...</code> includes every index of Arg
     * (thus calling for a complete reduction), the function may return either a scalar value or a one-by-one matrix.
     */
#ifdef __cpp_concepts
    template<std::size_t...indices, typename BinaryFunction, indexible Arg>
#else
    template<std::size_t...indices, typename BinaryFunction, typename Arg>
#endif
    static constexpr auto
    reduce(BinaryFunction&& op, Arg&& arg) = delete;


    /**
     * \brief Convert Arg to a set of coordinates in Euclidean space, based on \ref vector_space_descriptor C.
     * \note This is optional. If not defined, the public \ref OpenKalman::to_euclidean "to_euclidean" function
     * will construct a \ref ToEuclideanExpr object.
     */
#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C>
    static constexpr indexible auto
#else
    template<typename Arg, typename C>
    static constexpr auto
#endif
    to_euclidean(Arg&& arg, const C& c) = delete;


    /**
     * \brief Convert Arg from a set of coordinates in Euclidean space, based on \ref vector_space_descriptor C.
     * \note This is optional. If not defined, the public \ref OpenKalman::from_euclidean "from_euclidean" function
     * will construct a \ref FromEuclideanExpr object.
     */
#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C>
    static constexpr indexible auto
#else
    template<typename Arg, typename C>
    static constexpr auto
#endif
    from_euclidean(Arg&& arg, const C& c) = delete;


    /**
     * \brief Wrap Arg based on \ref vector_space_descriptor C.
     * \note This is optional. If not defined, the public \ref OpenKalman::wrap_angles "wrap_angles" function
     * will call <code>from_euclidean(to_euclidean(std::forward<Arg>(arg), c), c)</code>.
     */
#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor C>
    static constexpr indexible auto
#else
    template<typename Arg, typename C>
    static constexpr auto
#endif
    wrap_angles(Arg&& arg, const C& c) = delete;


    /**
     * \brief Take the conjugate of the argument.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     */
#ifdef __cpp_concepts
    template<indexible Arg>
    static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
    template<typename Arg>
    static constexpr auto
#endif
    conjugate(Arg&& arg) = delete;


    /**
     * \brief Take the transpose of the argument.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     */
#ifdef __cpp_concepts
    template<indexible Arg>
    static constexpr compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>> auto
#else
    template<typename Arg>
    static constexpr auto
#endif
    transpose(Arg&& arg) = delete;


    /**
     * \brief Take the adjoint of the argument.
     * \note This is optional. If not defined, the adjoint will be calculated as the transpose of the conjugate.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     */
#ifdef __cpp_concepts
    template<indexible Arg>
    static constexpr compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>> auto
#else
    template<typename Arg>
    static constexpr auto
#endif
    adjoint(Arg&& arg) = delete;


    /**
     * \brief Take the determinant of the argument.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     */
#ifdef __cpp_concepts
    template<square_shaped<Qualification::depends_on_dynamic_shape> Arg>
    static constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
    template<typename Arg>
    static constexpr auto
#endif
    determinant(Arg&& arg) = delete;


    /**
     * \brief Perform an element-by-element sum of compatible tensor-like objects
     * \note: An interface should at least define this for two arguments.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     * \param args Other \ref indexible objects of the same dimensions as Arg (but potentially from a different library).
     */
#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptors_may_match_with<Arg>...Args>
    static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
    template<typename Arg, typename...Args>
    static constexpr auto
#endif
    sum(Arg&& arg, Args&&...args) = delete;


    /**
     * \brief Multiple an object by a scalar value.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     * \param s A scalar value.
     * \note This is optional. If not defined, the library will use n_ary_operation with a constant.
     */
#ifdef __cpp_concepts
    template<indexible Arg, scalar_constant S> requires
      requires(S s) { {get_scalar_constant_value(s)} -> std::convertible_to<scalar_type_of_t<Arg>>; }
    static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
    template<typename Arg, typename S>
    static constexpr auto
#endif
    scalar_product(Arg&& arg, S&& s) = delete;


    /**
     * \brief Divide an object by a scalar value.
     * \param arg An \ref indexible object within the same library as LibraryObject.
     * \param s A scalar value.
     * \note This is optional. If not defined, the library will use n_ary_operation with a constant.
     */
#ifdef __cpp_concepts
    template<indexible Arg, scalar_constant S> requires
      requires(S s) { {get_scalar_constant_value(s)} -> std::convertible_to<scalar_type_of_t<Arg>>; }
    static constexpr vector_space_descriptors_may_match_with<Arg> auto
#else
    template<typename Arg, typename S>
    static constexpr auto
#endif
    scalar_quotient(Arg&& arg, S&& s) = delete;


    /**
     * \brief Perform a contraction involving two compatible tensors
     * \param a An \ref indexible object within the same library as LibraryObject
     * \param b Another \ref indexible object of the same dimensions as A (but potentially from a different library).
     */
#ifdef __cpp_concepts
    template<indexible A, vector_space_descriptors_may_match_with<A> B>
    static constexpr compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>> auto
#else
    template<typename A, typename B>
    static constexpr auto
#endif
    contract(A&& a, B&& b) = delete;


    /**
     * \brief Perform an in-place contraction involving two compatible tensors
     * \param a A \ref writable object within the same library as LibraryObject
     * \param b Another \ref indexible object of the same dimensions as A (but potentially from a different library).
     * \return A reference to A
     */
#ifdef __cpp_concepts
    template<bool on_the_right, writable A, vector_space_descriptors_may_match_with<A> B>
#else
    template<bool on_the_right, typename A, typename B>
#endif
    static constexpr A&
    contract_in_place(A& a, B&& b) = delete;


    /**
     * \brief Take the Cholesky factor of matrix Arg
     * \tparam triangle_type The \ref TriangleType of the result.
     * \param a An \ref indexible object within the same library as LibraryObject. It need not be hermitian, but
     * components outside the triangle defined by triangle_type will be ignored, and instead
     * \return A matrix t where tt<sup>T</sup> = a (if triangle_type == TriangleType::lower) or
     * t<sup>T</sup>t = a (if triangle_type == TriangleType::upper).
     */
#ifdef __cpp_concepts
    template<TriangleType triangle_type>
    static constexpr triangular_matrix<triangle_type> auto
    cholesky_factor(indexible auto&& a) = delete;
#else
    template<TriangleType triangle_type, typename Arg>
    static constexpr auto
    cholesky_factor(Arg&& a) = delete;
#endif


    /**
     * \brief Do a rank update on a hermitian matrix.
     * \note This is preferably (but not necessarily) performed as an in-place operation.
     * \details A must be a \ref hermitian_matrix.
     * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
     * - If A is a non-const lvalue reference, it should be updated in place if possible. Otherwise, the function may return a new matrix.
     * \tparam significant_triangle The triangle which is significant (or TriangleType::any if both are significant)
     * \param a A writable object (same library as type LibraryObject) in which triangle t is significant.
     * \param u The update vector or matrix.
     * Note: This may not necessarily be within the same library as a, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \returns an updated native, writable matrix in hermitian form.
     */
#ifdef __cpp_concepts
    template<HermitianAdapterType significant_triangle, indexible A, indexible U>
    static hermitian_matrix decltype(auto)
#else
    template<HermitianAdapterType significant_triangle, typename A, typename U>
    static decltype(auto)
#endif
    rank_update_hermitian(A&& a, U&& u, const scalar_type_of_t<A>& alpha) = delete;


    /**
     * \brief Do a rank update on a triangular matrix.
     * \note This is preferably (but not necessarily) performed as an in-place operation.
     * \details A must be a triangular matrix.
     * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
     * returning the updated A.
     * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
     * - If A is a non-const lvalue reference, it should be updated in place if possible. Otherwise, the function may return a new matrix.
     * \tparam triangle The triangle (upper or lower)
     * \param a An object of type LibraryObject, which is either triangular or dense-writable.
     * \param u The update vector or matrix.
     * Note: This may not necessarily be within the same library as a, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \param alpha Factor α
     * \returns an updated native, writable matrix in triangular (or diagonal) form.
     */
#ifdef __cpp_concepts
    template<TriangleType triangle, indexible A, indexible U> requires
      (triangle == TriangleType::lower) or (triangle == TriangleType::upper)
    static triangular_matrix<triangle> decltype(auto)
#else
    template<TriangleType triangle, typename A, typename U>
    static decltype(auto)
#endif
    rank_update_triangular(A&& a, U&& u, const scalar_type_of_t<A>& alpha) = delete;


    /**
     * \brief Solve the equation AX = B for X, which may or may not be a unique solution.
     * \tparam must_be_unique Determines whether the function throws an exception if the solution X is non-unique
     * (e.g., if the equation is under-determined)
     * \tparam must_be_exact Determines whether the function throws an exception if it cannot return an exact solution,
     * such as if the equation is over-determined. * If <code>false<code>, then the function will return an estimate
     * instead of throwing an exception.
     * \param a The matrix A in the equation AX = B
     * \param b The matrix B in the equation AX = B
     * Note: This may not necessarily be within the same library as a, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \return The solution X of the equation AX = B. If <code>must_be_unique</code>, then the function can return
     * any valid solution for X.
     */
#ifdef __cpp_concepts
    template<bool must_be_unique = false, bool must_be_exact = false, indexible A, indexible B>
    static compatible_with_vector_space_descriptors<vector_space_descriptor_of_t<A, 1>, vector_space_descriptor_of_t<B, 1>> auto
    solve(indexible auto&& a, indexible auto&& b) = delete;
#else
    template<bool must_be_unique = false, bool must_be_exact = false, typename A, typename B>
    static auto
    solve(A&& a, B&& b) = delete;
#endif


    /**
     * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
     * \note This is optional and can be derived if QR_decomposition is defined.
     * \param arg The matrix A to be decomposed
     * \returns L as a lower \ref triangular_matrix
     */
#ifdef __cpp_concepts
    static constexpr triangular_matrix<TriangleType::lower> auto
    LQ_decomposition(indexible auto&& arg) = delete;
#else
    template<typename Arg>
    static constexpr auto
    LQ_decomposition(Arg&& arg) = delete;
#endif


    /**
     * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
     * \note This is optional and can be derived if LQ_decomposition is defined.
     * \param arg The matrix A to be decomposed
     * \returns U as an upper \ref triangular_matrix
     */
#ifdef __cpp_concepts
    static constexpr triangular_matrix<TriangleType::upper> auto
    QR_decomposition(indexible auto&& arg) = delete;
#else
    template<typename Arg>
    static constexpr auto
    QR_decomposition(Arg&& arg) = delete;
#endif

#endif // DOXYGEN_SHOULD_SKIP_THIS
  };


} // namespace OpenKalman::interface


#endif //OPENKALMAN_LIBRARY_INTERFACE_HPP
