/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "coordinates/coordinates.hpp"
#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/internal/layout_mapping_policy.hpp"

namespace OpenKalman::interface
{
  /**
   * \brief An interface to various routines from the linear algebra library associated with \ref indexible object T.
   * \details This traits class must be specialized for any object (matrix, tensor, etc.) from a linear algebra library.
   * Typically, only one specialization would be necessary for all objects within a given library.
   * \tparam T An \ref indexible object that is native to the linear algebra library of interest.
   * Normally, this is used simply to select the correct library for processing the arguments.
   * But in some cases, the interface may also base the result on the properties of T (e.g., whether it is a matrix or array).
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct library_interface
  {
    static_assert(std::is_object_v<T>);

#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /**
     * \brief The base class within the library of T for custom wrappers (optional).
     * \details This is used when a library requires custom objects to derive from a particular base class.
     * For example, the Eigen library requires objects to derive from classes such as <code>Eigen::EigenBase</code>,
     * <code>Eigen::MatrixBase</code>, or <code>Eigen::ArrayBase</code>.
     * The particular base class within a given library can depend on the type of T
     * (e.g., whether T is a matrix or array).
     */
    template<typename Derived>
    using library_base = std::monostate;


    /**
     * \brief Get a scalar component of Arg at a given set of indices.
     * \details The indices are in the form of a \ref collections::index "index collection".
     * \param t An instance of T.
     * \param indices An \ref index_collection_for T.
     * \note Mandatory. Also, this function, or the library, is responsible for any optional bounds checking.
     *
     */
    static constexpr auto
    get_component = [](const T& t, index_collection_for<T> const auto& indices)
      -> std::same_as<element_type_of_t<T>> decltype(auto)
    {
      return 0;
    };


    /**
     * \brief Set a component of Arg at a given set of indices to scalar value s.
     * \details The indices are are in the form of a ranged object accessible by an iterator.
     * \tparam Indices A ranged object satisfing std::ranges::input_range, which contains exactly
     * <code>count_indices(arg)</code> indices.
     * \note Mandatory. Also, this function, or the library, is responsible for any optional bounds checking.
     *
     */
    static constexpr auto
    set_component = [](T& t, const index_collection_for<T> auto& indices, const element_type_of_t<T>& s) -> void
    {};


    /**
     * \brief Converts Arg (if it is not already) to a native matrix operable within the library associated with T.
     * \details The result should be in a form for which basic matrix operations can be performed within the library for T.
     * This should be a lightweight transformation that does not require copying of elements.
     * If possible, properties such as \ref diagonal_matrix, \ref triangular_matrix, \ref hermitian_matrix,
     * \ref constant_matrix, and \ref constant_diagonal_matrix should be preserved in the resulting object.
     * \note if not defined, a call to \ref OpenKalman::to_native_matrix will construct a \ref LibraryWrapper.
     */
    static constexpr auto
    to_native_matrix = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> indexible decltype(auto)
    {
      return std::forward<Arg>(arg);
    };


    /**
     * \brief Assign (by copying or moving) the elements of an indexible object to another indexible object.
     * \param other The \ref indexible object from which to assign.
     * \note Optional. If omitted, the \ref OpenKalman::assign function will attempt to assign in other ways.
     */
    static constexpr auto
    assign = [](T& t, indexible auto&& other) -> void
    {
      t = std::forward<decltype(other)>(other);
    };


    /**
     * \brief Make a default, potentially uninitialized, dense, writable matrix or array within the library.
     * \details Takes a \ref coordinates::euclidean_pattern_collection that specifies the dimensions of the resulting object
     * \tparam layout the \ref layout_mapping_policy of the result, which may be either std::layout_left or std::layout_right.
     * \tparam Scalar The scalar value of the result.
     * \return A default, potentially uninitialized, dense, writable object.
     * \note The interface may base the return value on any properties of T (e.g., whether T is a matrix or array).
     */
    template<internal::layout_mapping_policy layout, values::number Scalar> requires
      std::same_as<layout, stdcompat::layout_left> or std::same_as<layout, stdcompat::layout_right>
    static constexpr auto
    make_default = [](const coordinates::euclidean_pattern_collection auto& descriptors) -> indexible auto
    {
      // ...
    };


    /**
     * \brief Fill a writable matrix with a list of elements.
     * \tparam layout The \ref layout_mapping_policy of the listed elements, which may be std::layout_left or std::layout_right.
     * \tparam t The \ref writable object to be filled.
     * \param scalars A flat collection of values representing the elements. There must be exactly the right number of elements
     * to fill Arg.
     */
    template<internal::layout_mapping_policy layout> requires
      (std::same_as<layout, stdcompat::layout_left> or std::same_as<layout, stdcompat::layout_right>) and writable<T>
    static constexpr auto
    fill_components = []<typename Scalars> requires std::convertible_to<collections::common_collection_type_t<Scalars>, element_type_of_t<T>>
      (T& t, const collections::collection auto& scalars) -> void
    {
      // ...
    };


    /**
     * \brief Create a \ref constant_matrix of a given shape (optional).
     * \details Takes a \ref coordinates::euclidean_pattern_collection that specifies the dimensions of the resulting object
     * \param c A \ref values::scalar (either static or dynamic)
     * \param d A \ref coordinates::euclidean_pattern_collection
     * \note If this is not defined, calls to <code>OpenKalman::make_constant</code> will return an object of type constant_adapter.
     */
    static constexpr auto
    make_constant = [](const values::scalar auto& c, coordinates::euclidean_pattern_collection auto&& d)
      -> constexpr_matrix auto
    {
      // ...
    };


    /**
     * \brief Create a generalized \ref identity_matrix of a given shape (optional).
     * \details This is a generalized identity matrix that need not be square, but every non-diagonal element must be zero.
     * \note If not defined, an identity matrix is a \ref diagonal_adapter with a constant diagonal of 1.
     * \tparam Scalar The scalar type of the new object
     * \param d A \ref coordinates::pattern object defining the size
     */
    template<values::number Scalar>
    static constexpr auto
    make_identity_matrix = [](coordinates::euclidean_pattern_collection auto&& d) -> identity_matrix auto
    {
      // ...
    };


    /**
     * \brief Create a \ref triangular_matrix from a square matrix.
     * \details This is used by the function OpenKalman::make_triangular_matrix. This can be left undefined if
     * - Arg is already triangular and of a triangle_type compatible with t, or
     * - the intended result is for Arg to be wrapped in an \ref Eigen::TriangularAdapter (which will happen automatically).
     * \tparam tri The intended \ref triangle_type of the result.
     * \param arg A square matrix to be wrapped in a triangular adapter.
     */
    template<triangle_type tri>
    static constexpr auto
    make_triangular_matrix = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> triangular_matrix<tri> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Make a hermitian adapter.
     * \details This is used by the function OpenKalman::make_hermitian_matrix. This can be left undefined if
     * - Arg is already hermitian and of a HermitianAdapterType compatible with t, or
     * - the intended result is for Arg to be wrapped in an \ref Eigen::HermitianAdapter (which will happen automatically).
     * \tparam h The intended \ref HermitianAdapterType of the result.
     * \param arg A square matrix to be wrapped in a hermitian hermitian.
     */
    template<HermitianAdapterType h>
    static constexpr auto
    make_hermitian_adapter = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> hermitian_matrix<h> decltype(auto)
    {
      // ...
    };


     /**
      * \brief Project the vector space associated with index 0 to a Euclidean space for applying directional statistics.
      * \note This is optional.
      * If not defined, the \ref OpenKalman::to_euclidean "to_euclidean" function will construct a ToEuclideanExpr object.
      * In this case, the library should be able to accept the ToEuclideanExpr object as native.
      */
    static constexpr auto
    to_euclidean = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> has_untyped_index<0> decltype(auto)
    {
      // ...
    };


     /**
      * \brief Project the Euclidean vector space associated with index 0 to \ref coordinates::pattern v after applying directional statistics
      * \param v The new \ref coordinates::pattern for index 0.
      * \note This is optional.
      * If not defined, the \ref OpenKalman::from_euclidean "from_euclidean" function will construct a FromEuclideanExpr object.
      * In this case, the library should be able to accept FromEuclideanExpr object as native.
      */
    static constexpr auto
    from_euclidean = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T> and all_fixed_indices_are_euclidean<T>
      (Arg&& arg, const has_untyped_index<0> auto& v) -> indexible decltype(auto)
    {
      // ...
    };


     /**
      * \brief Wrap Arg based on \ref coordinates::pattern V.
      * \note This is optional. If not defined, the public \ref OpenKalman::wrap_angles "wrap_angles" function
      * will call <code>from_euclidean(to_euclidean(std::forward<Arg>(arg)), get_pattern_collection<0>(arg))</code>.
      */
    static constexpr auto
    wrap_angles = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> std::convertible_to<const T&> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Get a block from a matrix or tensor.
     * \param offsets An \ref collections::index_collection "index_collection" specifying the offsets to the slice.
     * \param extents An \ref collections::index_collection "index_collection" specifying the extents of the slice.
     */
    static constexpr auto
    get_slice = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg, const index_collection_for<Arg> auto& begin, const collections::index_collection auto& size)
      -> indexible decltype(auto)
    {
      // ...
    };


    /**
     * \brief Set a block from a \ref writable matrix or tensor.
     * \param t The matrix or tensor to be modified.
     * \param block A block to be copied into Arg at a particular location.
     * Note: This may not necessarily be within the same library as arg, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \param offsets \ref collections::index specifying the beginning \ref values::index.
     */
    static constexpr auto
    set_slice = [] requires writable<T>
      (T& t, indexible auto&& block, const index_collection_for<T> auto& offsets) -> void
    {
      // ...
    };


    /**
     * \brief Set only a triangular (or diagonal) portion of a \ref writable matrix with elements of another matrix.
     * \details Neither a nor b need to be square matrices.
     * \note This is optional.
     * \tparam tri The triangle_type (upper, lower, or diagonal)
     * \param a The matrix or tensor to be set
     * \param b A matrix or tensor to be copied from, which may or may not be triangular.
     */
    template<triangle_type tri>
    static constexpr auto
    set_triangle = [] requires writable<T>
      (T& t, vector_space_descriptors_may_match_with<T> auto&& b) -> void
    {
      // ...
    };


    /**
     * \brief Convert a column vector (or column slice for rank 2+ tensors) into a diagonal matrix (optional).
     * \note If this is not defined, calls to <code>OpenKalman::to_diagonal</code> will construct a \ref diagonal_adapter.
     * \details An interface need not deal with an object known to be \ref one_dimensional at compile time.
     * \tparam Arg An \ref indexible object with one higher rank than the argument
     */
    static constexpr auto
    to_diagonal = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> diagonal_matrix auto
    {
      // ...
    };


    /**
     * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
     * \details The argument need not be a square matrix.
     * An interface need not deal with the following situations, which are already handled by the
     * global \ref OpenKalman::diagonal_of "diagonal_of" function:
     * - an identity matrix
     * - a zero matrix
     * - a constant matrix or constant-diagonal matrix
     * \param arg An \ref indexible object with one lower rank than the argument (unless the rank is already 0)
     */
    static constexpr auto
    diagonal_of = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> indexible auto
    {
      // ...
    };


    /**
     * \brief Broadcast an object by replicating it by factors specified for each index.
     * \details The operation may increase the order of the object by specifying factors beyond the order of the argument.
     * \param arg The object.
     * \param factors A set of factors indicating the increase in size of each index. There must be one factor per
     * index, and there may also be additional factors if the tensor order is to be expanded
     */
    static constexpr auto
    broadcast = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg, const collections::index_collection auto& factors) -> indexible auto
    {
      // ...
    };


    /**
     * \brief Perform an n-ary array operation on a set of n arguments.
     * \details The \ref coordinates::pattern tuple d_tup defines the size of the resulting matrix.
     * \note This is optional and should be left undefined to the extent the native library does not provide this
     * functionality.
     * \param patterns A collection of \ref coordinates::pattern objects defining the resulting tensor
     * \tparam Operation The n-ary operation taking an optional collection of indices and n scalar arguments.
     * Examples:
     * - <code>template<values::number...X> operation(const X&...)</code>
     * - <code>template<values::number...X> operation(std::array<std::size_t, collections::size_of_v<Patterns>>, const X&...)</code>
     * - <code>template<values::number...X> operation(std::vector<std::size_t>, const X&...)</code>
     * \param args A set of n indexible arguments, each having a pattern consistent with <code>patterns</code>.
     * \return An object with size and shape defined by <code>patterns</code> and with elements defined by the operation
     */
    static constexpr auto
    n_ary_operation = []<coordinates::pattern_collection Patterns, typename Operation,
        compatible_with_vector_space_descriptor_collection<Patterns>...Args>
      requires
        std::invocable<Operation&&, element_type_of_t<Args>...> or
        std::invocable<Operation&&, std::array<std::size_t, collections::size_of_v<Patterns>>, element_type_of_t<Args>...> or
        std::invocable<Operation&&, std::vector<std::size_t>, element_type_of_t<Args>...>
      (const Patterns& patterns, Operation&& op, Args&&...args)
        -> compatible_with_vector_space_descriptor_collection<Patterns> auto
    {
      // ...
    };


    /**
     * \brief Use a binary function to reduce a tensor across one or more of its indices.
     * \details The binary function is assumed to be associative, so any order of operation is permissible.
     * \tparam indices The indices to be reduced. There will be at least one index. The order does not matter.
     * \param binary_function A binary function invocable with two values of type <code>element_type_of_t<Arg></code>
     * (e.g. std::plus, std::multiplies)
     * \param arg An object to be reduced
     * \returns A vector or tensor with reduced dimensions. If <code>indices...</code> includes every index of Arg
     * (thus calling for a complete reduction), the function may return either a scalar value or a one-by-one matrix.
     */
    template<std::size_t...indices, typename BinaryFunction>
    static constexpr auto
    reduce = []<typename Arg, typename BinaryFunction>
      requires std::same_as<std::remove_cvref_t<Arg>, T> and
        std::invocable<BinaryFunction, element_type_of_t<Arg>, element_type_of_t<Arg>>
      (Arg&& arg, BinaryFunction&& binary_function)
    {
      // ...
    };


    /**
     * \brief Take the conjugate of the argument.
     * \param arg An \ref indexible object within the same library as T.
     */
    static constexpr auto
    conjugate = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> constexpr vector_space_descriptors_may_match_with<Arg>
    {
      // ...
    };


    /**
     * \brief Take the transpose of the argument.
     * \param arg An \ref indexible object within the same library as T.
     */
    static constexpr auto
    transpose = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> dimension_size_of_index_is<0, index_dimension_of_v<T, 1>, applicability::permitted> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Take the adjoint of the argument.
     * \note This is optional. If not defined, the adjoint will be calculated as the transpose of the conjugate.
     * \param arg An \ref indexible object within the same library as T.
     */
    static constexpr auto
    adjoint = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> dimension_size_of_index_is<0, index_dimension_of_v<T, 1>, applicability::permitted> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Take the determinant of the argument.
     * \param arg An \ref indexible object within the same library as T.
     */
    static constexpr auto
    determinant = [] requires square_shaped<T, applicability::permitted>
      (const T& t) -> std::convertible_to<element_type_of_t<T>> auto
    {
      // ...
    };


    /**
     * \brief Perform an element-by-element sum of compatible tensor-like objects
     * \note: An interface should at least define this for two arguments.
     * \param args A set of \ref indexible objects.
     */
    static constexpr auto
    sum = []<typename...Args> requires vector_space_descriptors_may_match_with<Args...>
      (Args&&...args) -> vector_space_descriptors_may_match_with<Args...> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Multiple an object by a scalar value.
     * \param s A scalar value.
     * \note This is optional. If not defined, the library will use n_ary_operation with a constant.
     */
    static constexpr auto
    scalar_product = [](const T& t, std::convertible_to<element_type_of_t<T>> const auto& s)
      -> vector_space_descriptors_may_match_with<T> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Divide an object by a scalar value.
     * \param s A scalar value.
     * \note This is optional. If not defined, the library will use n_ary_operation with a constant.
     */
    static constexpr auto
    scalar_quotient = [](const T& t, std::convertible_to<element_type_of_t<T>> const auto& s)
      -> vector_space_descriptors_may_match_with<T> decltype(auto)
    {
      // ...
    };


    /**
     * \brief Perform a contraction involving two compatible tensors
     * \param a An \ref indexible object within the same library as T
     * \param b Another \ref indexible object of the same dimensions as A (but potentially from a different library).
     */
    static constexpr auto
    contract = []
      (const T& a, const dimension_size_of_index_is<0, index_dimension_of<T, 1>, applicability::permitted> auto& b)
      -> dimension_size_of_index_is<0, index_dimension_of_v<T, 0>, applicability::permitted> auto
    {
      // ...
    };


    /**
     * \brief Perform an in-place contraction involving two compatible tensors
     * \param a A \ref writable object within the same library as T
     * \param b Another \ref indexible object of the same dimensions as A (but potentially from a different library).
     */
    template<bool on_the_right>
    static constexpr auto
    contract_in_place = [](T& t, vector_space_descriptors_may_match_with<T> auto&& b) -> void
    {
      // ...
    };


    /**
     * \brief Take the Cholesky factor of matrix Arg
     * \tparam tri The \ref triangle_type of the result.
     * \param t An object of type T. It need not be hermitian, but
     * components outside the triangle defined by tri will be ignored, and instead
     * \return A matrix t where tt<sup>T</sup> = a (if tri == triangle_type::lower) or
     * t<sup>T</sup>t = a (if tri == triangle_type::upper).
     */
    template<triangle_type tri>
    static constexpr auto
    cholesky_factor = [](const T& t) -> triangular_matrix<tri> auto
    {
      // ...
    };


    /**
     * \brief Do a rank update on a hermitian matrix.
     * \note This is preferably (but not necessarily) performed in place.
     * \details A must be a \ref hermitian_matrix.
     * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
     * - If A is a non-const lvalue reference, it should be updated in place if possible. Otherwise, the function may return a new matrix.
     * \tparam significant_triangle The triangle which is significant (or triangle_type::any if both are significant)
     * \param a A writable object (same library as type T) in which triangle t is significant.
     * \param u The update vector or matrix.
     * Note: This may not necessarily be within the same library as a, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \returns an updated native, writable matrix in hermitian form.
     */
    template<HermitianAdapterType significant_triangle>
    static constexpr auto
    rank_update_hermitian = []<typename A, typename U>
      requires std::same_as<std::remove_cvref_t<A>, T> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, applicability::permitted> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, applicability::permitted> and
        std::convertible_to<element_type_of_t<U>, const element_type_of_t<A>>
      (A&& a, const U& u, const element_type_of_t<A>& alpha)
      -> hermitian_matrix decltype(auto)
    {
      // ...
    };


    /**
     * \brief Do a rank update on a triangular matrix.
     * \note This is preferably (but not necessarily) performed as an in-place operation.
     * \details A must be a triangular matrix.
     * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
     * returning the updated A.
     * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
     * - If A is a non-const lvalue reference, it should be updated in place if possible. Otherwise, the function may return a new matrix.
     * \tparam triangle The triangle (upper or lower)
     * \param a An object of type T, which is either triangular or dense-writable.
     * \param u The update vector or matrix.
     * Note: This may not necessarily be within the same library as a, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \param alpha Factor α
     * \returns an updated native, writable matrix in triangular (or diagonal) form.
     */
    template<triangle_type triangle> requires (triangle == triangle_type::lower) or (triangle == triangle_type::upper)
    static constexpr auto
    rank_update_triangular = []<typename A, typename U>
      requires std::same_as<std::remove_cvref_t<A>, T> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, applicability::permitted> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, applicability::permitted> and
        std::convertible_to<element_type_of_t<U>, const element_type_of_t<A>>
      (A&& a, const U& u, const element_type_of_t<decltype(a)>& alpha)
      -> triangular_matrix<triangle> decltype(auto)
    {
      // ...
    };


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
    template<bool must_be_unique = false, bool must_be_exact = false>
    static constexpr auto
    solve = []
      (const T& a, const dimension_size_of_index_is<0, index_dimension_of_v<A, 0>, applicability::permitted> auto& b)
        -> indexible auto
    {
      // ...
    };


    /**
     * \brief Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
     * \note This is optional and can be derived if QR_decomposition is defined.
     * \param arg The matrix A to be decomposed
     * \returns L as a lower \ref triangular_matrix
     */
    static constexpr auto
    LQ_decomposition = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> triangular_matrix<triangle_type::lower> auto
    {
      // ...
    };


    /**
     * \brief Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
     * \note This is optional and can be derived if LQ_decomposition is defined.
     * \param arg The matrix A to be decomposed
     * \returns U as an upper \ref triangular_matrix
     */
    static constexpr auto
    QR_decomposition = []<typename Arg> requires std::same_as<std::remove_cvref_t<Arg>, T>
      (Arg&& arg) -> triangular_matrix<triangle_type::upper> auto
    {
      // ...
    };

#endif // DOXYGEN_SHOULD_SKIP_THIS
  };


}


#endif
