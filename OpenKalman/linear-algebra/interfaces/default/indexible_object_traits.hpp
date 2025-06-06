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
 * \brief Forward declaration of \ref indexible_object_traits, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP
#define OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP

#include <type_traits>
#include <tuple>

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to traits of a particular indexible object (i.e., a matrix or generalized tensor).
   * \details This traits class must be specialized for any \ref indexible object (matrix, tensor, etc.)
   * from a linear algebra library. Each different type of objects in a library will typically have its own specialization.
   * \tparam T An object, such as a matrix, array, or tensor, with components addressable by indices.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename>
#endif
  struct indexible_object_traits
  {
#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /**
     * \brief The scalar type of T (e.g., double, int).
     * \details <code>OpenKalman::values::number&lt;scalar_type&gt;</code> must be satisfied.
     * \note Mandatory.
     * \sa scalar_type_of
     */
    using scalar_type = double;


    /**
     * \brief Get the number of indices needed to access the elements of T (preferably as an std::integral_constant).
     * \details This generally corresponds to the tensor order. For example, this value would be 1
     * (preferably std::integral_constant<std::size_t, 1>{}) for a row vector and 2
     * (preferably std::integral_constant<std::size_t, 2>{}) for a matrix.
     * If a library allows more than one choice for the number of indices,
     * this value should be the maximum such value. For example, if a column vector is accessible by either
     * one or two indices, the value should be 2 (preferably std::integral_constant<std::size_t, 2>{}).
     * \note Mandatory. The \ref indexible concept applies iff this function is defined and returns an \ref values::index.
     * \return An \ref values::index representing the number of indices.
     * \sa OpenKalman::index_count
     * \sa OpenKalman::count_indices
     */
#ifdef __cpp_concepts
    static constexpr values::index auto
#else
    static constexpr auto
#endif
    count_indices(const T& arg) = delete;


    /**
     * \brief Get the \ref coordinates::pattern for index N of an argument.
     * \details The implementation may assume that <code>n &lt; count_indices(t)</code>.
     * \note Mandatory. The simplest implementation is to return the dimension of index N as <code>std::size_t</code>
     * (if dynamic) or as <code>std::integral_constant&gt;std::size_t, ...&gt;</code> (if static).
     * \param n An index value less than the result of \ref count_indices
     * (e.g., 0 (dynamic) or <code>std::integral_constant&lt;std::size_t, 0&gt;</code> (static))
     * \return A \ref coordinates::pattern object (either static or dynamic) for dimension <code>n</code>.
     */
#ifdef __cpp_concepts
    static constexpr coordinates::pattern auto
    get_vector_space_descriptor(const T& arg, const values::index auto& n) = delete;
#else
    template<typename N, std::enable_if_t<values::index<N>, int> = 0>
    static constexpr auto
    get_vector_space_descriptor(const T& arg, const N& n) = delete;
#endif


    /**
     * \brief Gets the nested object for T, if it exists.
     * /detail This should only be defined if T has a nested matrix.
     * \sa OpenKalman::nested_object
     */
#ifdef __cpp_concepts
    static indexible decltype(auto)
    nested_object(std::convertible_to<const T&> auto&& arg) = delete;
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const T&>, int> = 0>
    static decltype(auto)
    nested_object(Arg&& arg) = delete;
#endif


    /**
     * \brief If all components of the argument share the same constant value, return that constant.
     * \note: Optional.
     * \returns A \ref values::scalar (or, if no constant, some empty class such as std::monostate).
     */
#ifdef __cpp_concepts
    static constexpr values::scalar auto
#else
    static constexpr auto
#endif
    get_constant(const T& arg) = delete;


    /**
     * \brief If the argument is a \ref diagonal_matrix and all diagonal components share the same constant value, return that constant.
     * \details If T is a rank >2 tensor, every rank-2 slice comprising dimensions 0 and 1 must be constant diagonal matrix.
     * \note: Optional.
     * \returns A \ref values::scalar (or, if no constant diagonal, some empty class such as std::monostate).
     */
#ifdef __cpp_concepts
    static constexpr values::scalar auto
#else
    static constexpr auto
#endif
    get_constant_diagonal(const T& arg) = delete;


    /**
     * \brief Whether T is \ref one-dimensional in all indices.
     * \note: Optional. If omitted, T's status as one-by-one can usually be derived from the dimensions.
     * \details This can be useful because some types may erase information about the shape of their nested objects,
     * such as in an object that replicates an existing object in one or more directions.
     */
    template<Applicability b>
    static constexpr bool
    one_dimensional = false;


    /**
     * \brief Whether all dimensions of T are the same and type-equivalent (optional).
     * \note: Optional. If omitted, T's status as square can usually be derived from the dimensions.
     * \details This can be useful because some types may erase information about the shape of their nested objects.
     */
    template<Applicability b>
    static constexpr bool
    is_square = false;


    /**
     * \brief Whether T is triangular or diagonal with a particular \ref TriangleType.
     * \note Optional. Defaults to false if omitted.
     * \details This trait should propagate from any nested matrices or matrices involved in any expression arguments.
     * \tparam t The \ref TriangleType
     */
    template<TriangleType t>
    static constexpr bool
    is_triangular = false;


    /**
     * \brief Whether T is a \ref triangular_adapter.
     * \note Optional. Defaults to false if omitted.
     */
    static constexpr bool
    is_triangular_adapter = false;


    /**
     * \brief Whether T is hermitian.
     * \note Optional. If omitted, T is not considered hermitian.
     * \details This is unnecessary if T is a square matrix and any of the following are true:
     * - <code>scalar_type_of_t&lt;T&rt;</code> is real (not complex), or
     * - T is a \ref constant_matrix or \ref constant_diagonal_matrix in which the constant is real
     */
    static constexpr bool
    is_hermitian = false;


    /**
     * \brief If T is a \ref hermitian adapter, this specifies its \ref HermitianAdapterType.
     * \details A hermitian adapter is a (potentially) hermitian matrix formed by nesting a non-hermitian matrix within
     * an adapter that copies conjugated elements from the lower to upper triangle, or vice versa.
     * \note Optional. If omitted, T is not considered to be hermitian adapter.
     * It is also meaningless if \ref is_hermitian is false.
     */
    static constexpr HermitianAdapterType
    hermitian_adapter_type = HermitianAdapterType::any;


    /**
     * \brief Whether T is a writable, self-contained matrix or array.
     */
    static constexpr bool
    is_writable = false;


    /**
     * \brief If the argument has direct access to the underlying array data, return a pointer to that raw data.
     * \details This could be a
     */
#ifdef __cpp_lib_concepts
    template<std::convertible_to<const T&> Arg> requires requires(Arg&& arg) { {*std::forward<Arg>(arg).data()} -> values::scalar; } and direct_access
    static constexpr values::number decltype(auto)
#else
    template<typename Arg, std::enable_if_t<
      std::is_convertible_v<Arg, const T> and values::scalar<decltype(*std::declval<Arg&&>().data())> and direct_access, int> = 0>
    static constexpr auto decltype(auto)
#endif
    raw_data(Arg&& arg) = delete;


    /**
     * \brief The layout of T.
     */
    static constexpr Layout
    layout = Layout::none;


    /**
     * \brief If layout is Layout::stride, this returns a tuple of strides, one for each dimension.
     * \details This is only necessary if layout == Layout::stride. An interface can still provide strides for
     * \ref Layout::left and \ref Layout::right, and these will be used; however, this is not necessary and strides
     * can be calculated without this interface.
     * The tuple elements may be integral constants if the values are known at compile time. Example:
     * <code>return std::tuple {std::ptrdiff_t{16}, std::ptrdiff_t{4}, std::integral_constant<std::ptrdiff_t, 1>{}></code>
     * The stride values should be of type <code>std::ptrdiff_t</code> or <code>std::integral_constant<std::ptrdiff_t, ...></code>.
     * \return A tuple-like object of types std::ptrdiff_t or std::integral_constant<std::ptrdiff_t, ...>.
     */
    static constexpr auto
    strides(const T& arg) = delete;

#endif // DOXYGEN_SHOULD_SKIP_THIS
  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP
