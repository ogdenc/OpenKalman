/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#ifdef DOXYGEN_SHOULD_SKIP_THIS
#include "patterns/patterns.hpp"
#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/concepts/internal/layout_mapping_policy.hpp"
#include "linear-algebra/concepts/internal/slice_specifier.hpp"
#endif

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
      (T& t, patterns_may_match_with<T> auto&& b) -> void
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
     * \details The \ref patterns::pattern tuple d_tup defines the size of the resulting matrix.
     * \note This is optional and should be left undefined to the extent the native library does not provide this
     * functionality.
     * \param patterns A collection of \ref patterns::pattern objects defining the resulting tensor
     * \tparam Operation The n-ary operation taking an optional collection of indices and n scalar arguments.
     * Examples:
     * - <code>template<values::number...X> operation(const X&...)</code>
     * - <code>template<values::number...X> operation(std::array<std::size_t, collections::size_of_v<Patterns>>, const X&...)</code>
     * - <code>template<values::number...X> operation(std::vector<std::size_t>, const X&...)</code>
     * \param args A set of n indexible arguments, each having a pattern consistent with <code>patterns</code>.
     * \return An object with size and shape defined by <code>patterns</code> and with elements defined by the operation
     */
    static constexpr auto
    n_ary_operation = []<patterns::pattern_collection Patterns, typename Operation,
        compares_with_pattern_collection<Patterns>...Args>
      requires
        std::invocable<Operation&&, element_type_of_t<Args>...> or
        std::invocable<Operation&&, std::array<std::size_t, collections::size_of_v<Patterns>>, element_type_of_t<Args>...> or
        std::invocable<Operation&&, std::vector<std::size_t>, element_type_of_t<Args>...>
      (const Patterns& patterns, Operation&& op, Args&&...args)
        -> compares_with_pattern_collection<Patterns> auto
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
     * \brief Take the determinant of the argument.
     * \param arg An \ref indexible object within the same library as T.
     */
    static constexpr auto
    determinant = [] requires square_shaped<T, 2, applicability::permitted>
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
    sum = []<typename...Args> requires patterns_may_match_with<Args...>
      (Args&&...args) -> patterns_may_match_with<Args...> decltype(auto)
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
      -> patterns_may_match_with<T> decltype(auto)
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
      -> patterns_may_match_with<T> decltype(auto)
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
      (const T& a, const dimension_size_of_index_is<0, index_dimension_of<T, 1>, &stdex::is_eq, applicability::permitted> auto& b)
      -> dimension_size_of_index_is<0, index_dimension_of_v<T, 0>, &stdex::is_eq, applicability::permitted> auto
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
    contract_in_place = [](T& t, patterns_may_match_with<T> auto&& b) -> void
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
     * \tparam significant_triangle The triangle which is significant
     * \param a A writable object (same library as type T) in which triangle t is significant.
     * \param u The update vector or matrix.
     * Note: This may not necessarily be within the same library as a, so a conversion may be necessary
     * (e.g., via /ref to_native_matrix).
     * \returns an updated native, writable matrix in hermitian form.
     */
    template<triangle_type significant_triangle>
    static constexpr auto
    rank_update_hermitian = []<typename A, typename U>
      requires std::same_as<std::remove_cvref_t<A>, T> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, &stdex::is_eq, applicability::permitted> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, &stdex::is_eq, applicability::permitted> and
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
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 0>, &stdex::is_eq, applicability::permitted> and
        dimension_size_of_index_is<U, 0, index_dimension_of_v<A, 1>, &stdex::is_eq, applicability::permitted> and
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
      (const T& a, const dimension_size_of_index_is<0, index_dimension_of_v<A, 0>, &stdex::is_eq, applicability::permitted> auto& b)
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
