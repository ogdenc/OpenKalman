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
 * \brief Forward declaration of \ref indexible_object_traits, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP
#define OPENKALMAN_INDEXIBLE_OBJECT_TRAITS_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/enumerations.hpp"

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
  template<typename T, typename = void>
#endif
  struct indexible_object_traits
  {
    static_assert(std::is_object_v<T>);

#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /**
     * \brief Return an std::mdspan as a view to the object.
     * \note This is the only mandatory trait. The rest are optional.
     */
    static constexpr auto
    get_mdspan = [](T& t){ return std::mdspan{t}; };


    /**
     * \brief Get the \ref coordinates::pattern_collection associated with the object.
     * \note If omitted, T will be associated with a Euclidean pattern derived from the extents of the mdspan.
     */
    static constexpr auto
    get_pattern_collection = [](const T&) -> coordinates::pattern_collection auto { return std::tuple{}; };


    /**
     * \brief Gets the nested object for T, if it exists.
     * /detail This should only be defined if T has a nested matrix.
     * \sa OpenKalman::nested_object
     */
    static indexible decltype(auto)
    nested_object(std::convertible_to<const T&> auto&& arg) = delete;


    /**
     * \brief If all components of the argument share the same constant value, return that constant.
     * \note: Optional.
     * \returns A \ref values::scalar (or, if no constant, some empty class such as std::monostate).
     */
    static constexpr values::scalar auto
    get_constant(const T& arg) = delete;


    /**
     * \brief If the argument is a \ref diagonal_matrix and all diagonal components share the same constant value, return that constant.
     * \details If T is a rank >2 tensor, every rank-2 slice comprising dimensions 0 and 1 must be constant diagonal matrix.
     * \note: Optional.
     * \returns A \ref values::scalar (or, if no constant diagonal, some empty class such as std::monostate).
     */
    static constexpr values::scalar auto
    get_constant_diagonal(const T& arg) = delete;


    /**
     * \brief Whether T is \ref one-dimensional in all indices.
     * \note: Optional. If omitted, T's status as one-by-one can usually be derived from the dimensions.
     * \details This can be useful because some types may erase information about the shape of their nested objects,
     * such as in an object that replicates an existing object in one or more directions.
     */
    template<applicability b>
    static constexpr bool
    one_dimensional = false;


    /**
     * \brief Whether all dimensions of T are the same and type-equivalent (optional).
     * \note: Optional. If omitted, T's status as square can usually be derived from the dimensions.
     * \details This can be useful because some types may erase information about the shape of their nested objects.
     */
    template<applicability b>
    static constexpr bool
    is_square = false;


    /**
     * \brief Whether T is triangular or diagonal with a particular \ref triangle_type.
     * \note Optional. Defaults to false if omitted.
     * \details This trait should propagate from any nested matrices or matrices involved in any expression arguments.
     * \tparam t The \ref triangle_type
     */
    template<triangle_type t>
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
     * - <code>element_type_of_t&lt;T&rt;</code> is real (not complex), or
     * - T is a \ref constant_matrix or \ref constant_diagonal_matrix in which the constant is real
     */
    static constexpr bool
    is_hermitian = false;


    /**
     * \brief If T is a \ref hermitian adapter, this specifies its \ref HermitianAdapterType.
     * \details A hermitian adapter is a (potentially) hermitian matrix formed by nesting a non-hermitian matrix within
     * an adapter that copies conjugated elements from the lower to upper triangle, or vice versa.
     * \note Optional. If omitted, or if HermitianAdapterType::none, T is not considered to be hermitian adapter.
     * It is also meaningless if \ref is_hermitian is false.
     */
    static constexpr HermitianAdapterType
    hermitian_adapter_type = HermitianAdapterType::any;

#endif // DOXYGEN_SHOULD_SKIP_THIS
  };

}


#endif
