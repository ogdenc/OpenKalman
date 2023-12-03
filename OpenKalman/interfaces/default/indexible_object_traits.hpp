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
 * \brief Forward declaration of \ref indexible_object_traits, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP
#define OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP

#include <type_traits>
#include <tuple>


namespace OpenKalman::interface
{
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename>
#endif
  struct indexible_object_traits
  {
    /**
     * \brief The scalar type of T (e.g., double, int).
     * \details <code>OpenKalman::scalar_type&lt;scalar_type&gt;</code> must be satisfied.
     * \note Mandatory.
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
     * \note Mandatory. The \ref indexible concept applies iff this function is defined and returns an \ref index_value.
     * \return An \ref index_value (either \ref static_index_value or \ref dynamic_index_value) representing the number of indices.
     */
#ifdef __cpp_concepts
    static constexpr index_value auto
#else
    static constexpr auto
#endif
    get_index_count(const T& arg) = delete;


    /**
     * \brief Get the \ref vector_space_descriptor for index N of an argument.
     * \details The implementation may assume that <code>n &lt; get_index_count(t)</code>.
     * \note Mandatory. The simplest implementation is to return the dimension of index N as <code>std::size_t</code>
     * (if dynamic) or as <code>std::integral_constant&gt;std::size_t, ...&gt;</code> (if static).
     * \param n An index value less than the result of \ref get_index_count
     * (e.g., 0 (dynamic) or <code>std::integral_constant&lt;std::size_t, 0&gt;</code> (static))
     * \return A \ref vector_space_descriptor object (either static or dynamic) for dimension <code>n</code>.
     */
#ifdef __cpp_concepts
    static constexpr vector_space_descriptor auto
    get_vector_space_descriptor(const T& arg, index_value auto n) = delete;
#else
    template<typename N, std::enable_if_t<index_value<N>, int> = 0>
    static constexpr auto
    get_vector_space_descriptor(const T& arg, N n) = delete;
#endif


    /**
     * \typedef type
     * \brief A tuple with elements corresponding to each dependent object.
     * \details If the object is linked within T by an lvalue reference, the element should be an lvalue reference.
     * Examples:
     * \code
     *   using dependents = std::tuple<>; //< T has no dependencies
     *   using dependents = std::tuple<Arg1, Arg2&>; //< T stores Arg1 and a reference to Arg2
     * \endcode
     * \note Optional. If this is not defined, T will be considered non-self-contained.
     */
     using dependents = std::tuple<>;


    /**
     * \brief Indicates whether type T stores any internal runtime parameters.
     * \details An example of an internal runtime parameter might be indices for start locations, or sizes, for an
     * expression representing a block or sub-matrix within a matrix. If unknown, the value of <code>true</code> is
     * the safest and will prevent unintended dangling references.
     * \note Optional. If this is not defined, T will be treated as if it is defined and true. This parameter can ignore whether
     * any nested matrices, themselves, have internal runtime parameters.
     */
    static constexpr bool has_runtime_parameters = false;


    /**
     * \brief Gets the i-th dependency of T.
     * /detail There is no need to check the bounds of <code>i</code>, but they should be treated as following this
     * constraint:
     * /code
     *   requires (i < std::tuple_size_v<type>) and std::same_as<std::decay_t<Arg>, std::decay_t<T>>
     * /endcode
     * \note Optional. Also, there is no need for the example constraints on i or Arg,
     * as OpenKalman::nested_matrix already enforces these constraints.
     * \tparam i Index of the dependency (0 for the 1st dependency, 1 for the 2nd, etc.).
     * \return An \ref indexible object which is the i-th dependency of T.
     * \sa OpenKalman::nested_matrix
     */
#ifdef __cpp_concepts
    template<std::size_t i> static indexible decltype(auto)
    get_nested_matrix(std::convertible_to<const T&> auto&& arg) = delete;
#else
    template<std::size_t i, typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const T&>, int> = 0>
    static decltype(auto)
    get_nested_matrix(Arg&& arg) = delete;
#endif


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
    static indexible decltype(auto)
    convert_to_self_contained(std::convertible_to<const T&> auto&& arg) = delete;
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const T&>, int> = 0>
    static decltype(auto)
    convert_to_self_contained(Arg&& arg) = delete;
#endif


    /**
     * \brief If all components of the argument share the same constant value, return that constant.
     * \note: Optional.
     * \returns A \ref scalar_constant (or, if no constant, some empty class such as std::monostate).
     */
#ifdef __cpp_concepts
    static constexpr scalar_constant auto
#else
    static constexpr auto
#endif
    get_constant(const T& arg) = delete;


    /**
     * \brief If the argument is diagonal and all diagonal components share the same constant value, return that constant.
     * \note: Optional.
     * \returns A \ref scalar_constant (or, if no constant diagonal, some empty class such as std::monostate).
     */
#ifdef __cpp_concepts
    static constexpr scalar_constant auto
#else
    static constexpr auto
#endif
    get_constant_diagonal(const T& arg) = delete;


    /**
     * \brief Whether T is \ref one-dimensional in all indices.
     * \note: Optional. If omitted, T's status as one-by-one can usually be derived from the dimensions.
     * \details This can be useful because some types may erase information about the shape of their nested objects.
     */
    template<Likelihood b>
    static constexpr bool is_one_by_one = false;


    /**
     * \brief Whether all dimensions of T are the same and type-equivalent (optional).
     * \note: Optional. If omitted, T's status as square can usually be derived from the dimensions.
     * \details This can be useful because some types may erase information about the shape of their nested objects.
     */
    template<Likelihood b>
    static constexpr bool is_square = false;


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
     * \brief Whether T is hermitian.
     * \note Optional. If omitted, T is not considered hermitian.
     * \details This is unnecessary if T is a square matrix and any of the following are true:
     * - <code>scalar_type_of_t&lt;T&rt;</code> is real (not complex), or
     * - T is a \ref constant_matrix or \ref constant_diagonal_matrix in which the constant is real
     */
    static constexpr bool is_hermitian = false;


    /**
     * \brief If T is a \ref hermitian adapter, this specifies its \ref HermitianAdapterType.
     * \details A hermitian adapter is a (potentially) hermitian matrix formed by nesting a non-hermitian matrix within
     * an adapter that copies conjugated elements from the lower to upper triangle, or vice versa.
     * \note Optional. If omitted, T is not considered to be hermitian adapter.
     * It is also meaningless unless \ref is_hermitian is true.
     */
    static constexpr HermitianAdapterType hermitian_adapter_type = HermitianAdapterType::any;


    /**
     * \brief Whether T is a writable, self-contained matrix or array.
     */
    static constexpr bool is_writable = false;


    /**
     * \brief If the argument has direct access to the underlying array data, return a pointer to that data.
     */
#ifdef __cpp_concepts
    static constexpr std::convertible_to<const scalar_type * const> auto * const
    data(std::convertible_to<const T> auto& arg) = delete;
#else
    template<typename Arg, std::enable_if_t<std::is_convertible_v<Arg, const T>, int> = 0>
    static constexpr auto * const
    data(Arg& arg) = delete;
#endif


    /**
     * \brief The layout of T.
     */
    static constexpr Layout layout = Layout::none;


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

  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP
