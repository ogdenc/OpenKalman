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
 * \brief Forward declaration of \ref object_traits, which must be defined for all objects used in OpenKalman.
 */

#ifndef OPENKALMAN_OBJECT_TRAITS_HPP
#define OPENKALMAN_OBJECT_TRAITS_HPP

#ifdef DOXYGEN_SHOULD_SKIP_THIS
#include "coordinates/coordinates.hpp"
#include "linear-algebra/enumerations.hpp"
#endif

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
  struct object_traits
  {
    /**
     * \brief Identifies types for which object_traits is specialized.
     */
    static const bool
    is_specialized = false;


#ifdef DOXYGEN_SHOULD_SKIP_THIS
    /**
     * \brief Return an std::mdspan as a view to the object.
     * \note This is the only mandatory trait. The rest are optional.
     */
    static constexpr auto
    get_mdspan = [](std::convertible_to<const T&> auto&& t) -> decltype(auto) { return std::mdspan{t}; };


    /**
     * \brief Get the \ref coordinates::pattern_collection associated with the object.
     * \note Optional. If omitted, T will be associated with a Euclidean pattern derived from the extents of the mdspan.
     */
    static constexpr auto
    get_pattern_collection = [](std::convertible_to<const T&> auto&&)
      -> coordinates::pattern_collection decltype(auto) { return std::tuple{}; };


    /**
     * \brief The \ref triangle_type of the object (including \ref triangle_type::none "none" if it is not triangular.
     * \details This value cannot be \ref triangle_type::any.
     * This trait should propagate from any nested matrices or matrices involved in any expression arguments.
     * \note Optional. Defaults to \ref triangle_type::none if omitted.
     */
    static constexpr triangle_type
    triangle_type_value = triangle_type::none;


    /**
     * \brief If the components of the argument share the same constant value, return that constant.
     * \details If \ref triangle_type_value indicates that T has a triangular type, this reflects the constant
     * value of all non-zero components. For example, if <code>triangle_type_value == triangle_type::diagonal</code>,
     * this value is the value along the diagonal, with all other values being zero.
     * \note: Optional. If T is definitely not a constant object, this can be omitted or, alternatively,
     * this can return a non \ref values::value object such as std::monostate.
     */
    static constexpr auto
    get_constant = [](std::convertible_to<const T&> auto&&) -> values::value auto { return ...; };


    /**
     * \brief Gets the nested object for T, if it exists.
     * /detail This should only be defined if T has a nested matrix.
     * \note Optional. If T has no nested object, this must be omitted.
     * \sa OpenKalman::nested_object
     */
    static constexpr auto
    nested_object = [](std::convertible_to<const T&> auto&& t)
      -> indexible decltype(auto) { return std::forward<decltype(t)>(t); };


    /**
     * \brief Whether all dimensions of T are the same and type-equivalent (optional).
     * \note: Optional.
     * \details This is only necessary if the object is known to be square but the specific dimension is
     * unknown at compile time.
     */
    template<applicability b>
    static constexpr bool
    is_square = false;


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
     * - T is a \ref constant_object or \ref constant_diagonal_object in which the constant is real
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
