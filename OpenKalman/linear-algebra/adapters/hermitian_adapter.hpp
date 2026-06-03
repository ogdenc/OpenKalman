/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief hermitian_adapter and related definitions.
 */

#ifndef OPENKALMAN_HERMITIAN_ADAPTER_HPP
#define OPENKALMAN_HERMITIAN_ADAPTER_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/square_shaped.hpp"
#include "linear-algebra/traits/is_square_shaped.hpp"
#include "linear-algebra/concepts/triangular_matrix.hpp"
#include "linear-algebra/traits/constant_value_of.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/hermitian_accessor.hpp"
#include "linear-algebra/interfaces/stl/layout_triangle_partition.hpp"

namespace OpenKalman
{

  /**
   * \brief An adapter to create a \ref hermitian_matrix from a general matrix.
   * \details The matrix is guaranteed to be hermitian. Only elements in the storage_triangle are meaningful.
   * Other elements are derived from the storage triangle by complex conjugation.
   * Also, along the diagonal only the real part of the element is accessed.
   * \tparam Nested A nested \ref square_shaped object.
   * \tparam storage_triangle The triangle_type (\ref triangle_type::lower "lower"
   * or \ref triangle_type::upper "upper") in which the data is stored.
   */
#ifdef __cpp_concepts
  template<square_shaped<2, applicability::permitted> Nested, triangle_type storage_triangle> requires
    (storage_triangle == triangle_type::lower or storage_triangle == triangle_type::upper) and
    (not (constant_object<Nested> or constant_diagonal_object<Nested>) or
      not values::fixed<constant_value_of<Nested>> or
      values::not_complex<constant_value_of<Nested>>)
#else
  template<typename Nested, triangle_type storage_triangle>
#endif
  struct hermitian_adapter : OpenKalman::internal::adapter_base<hermitian_adapter<Nested, storage_triangle>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(square_shaped<Nested, 2, applicability::permitted>);
    static_assert(storage_triangle == triangle_type::lower or storage_triangle == triangle_type::upper);
    static_assert(not (constant_object<Nested> or constant_diagonal_object<Nested>) or
      not values::fixed<constant_value_of<Nested>> or
      values::not_complex<constant_value_of<Nested>>);
#endif

    using Base = OpenKalman::internal::adapter_base<hermitian_adapter, Nested>;

  public:

    /**
     * \brief Construct from the nested type.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires
      (not std::is_base_of_v<hermitian_adapter, std::decay_t<Arg>>) and
      std::constructible_from<Base, Arg&&>
#else
    template<typename Arg, std::enable_if_t<
      (not std::is_base_of_v<hermitian_adapter, std::decay_t<Arg>>) and
      stdex::constructible_from<Base, Arg&&>, int> = 0>
#endif
    constexpr explicit
    hermitian_adapter(Arg&& arg)
#ifdef __cpp_contracts
      pre (square_shaped<Arg, 2> or is_square_shaped<2>(arg))
#endif
      : Base {std::forward<Arg>(arg)}
    {
#ifndef __cpp_contracts
      if constexpr (not square_shaped<Arg, 2>) if (not is_square_shaped<2>(arg))
        throw (std::logic_error("Argument to hermitian_adapter is not square"));
#endif
    }

    /**
     * \brief Default constructor.
     */
    constexpr hermitian_adapter() = default;

  };


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    namespace detail
    {
      template<typename Nested, triangle_type tri>
      struct hermitian_object_traits
      {
        static const bool is_specialized = true;

        static constexpr triangle_type
        triangle_type_value =
          triangle_type_of_v<Nested> * tri == triangle_type::diagonal ? triangle_type::diagonal :
          triangle_type::none;

        template<std::size_t N, applicability b>
        static constexpr bool
        is_square = (N <= 2) or square_shaped<Nested, N, b>;

        static constexpr bool
        is_hermitian = true;
      };

    }


    template<typename Nested, triangle_type storage_type>
    struct object_traits<hermitian_adapter<Nested, storage_type>>
      : detail::hermitian_object_traits<Nested, storage_type>
    {
      static const bool is_specialized = true;

      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        auto n = OpenKalman::get_mdspan(std::forward<decltype(t)>(t).nested_object());
        using N = std::decay_t<decltype(n)>;
        using extents_type = typename N::extents_type;
        using nested_layout_type = typename N::layout_type;
        using nested_accessor_type = typename N::accessor_type;
        auto nested_m = n.mapping();

        using layout_type = layout_triangle_partition<nested_layout_type, storage_type>;
        using mapping_type = typename layout_type::template mapping<extents_type>;
        std::size_t partition_offset = nested_m.required_span_size();

        using accessor_type = hermitian_accessor<nested_accessor_type>;
        using data_handle_type = typename accessor_type::data_handle_type;

        return stdex::mdspan(
          data_handle_type {n.data_handle(), partition_offset},
          mapping_type {nested_m},
          accessor_type {n.accessor()});
      };

      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return OpenKalman::get_pattern_collection(std::forward<decltype(t)>(t).nested_object());;
      };

    private:

      static constexpr bool constant_real =
        (constant_object<Nested> or constant_diagonal_object<Nested>) and
        values::not_complex<constant_value_of<Nested>>;

    public:


#ifdef __cpp_concepts
      template<typename T> requires constant_real
#else
      template<typename T, bool Enable = true, std::enable_if_t<Enable and constant_real, int> = 0>
#endif
      static constexpr auto
      get_constant(T&& t)
      {
        return constant_value(std::forward<T>(t).nested_object());
      };

    };


    template<typename N, typename E, typename NL, triangle_type tri, typename NA>
    struct object_traits<stdex::mdspan<N, E, layout_triangle_partition<NL, tri>, hermitian_accessor<NA>>>
      : detail::hermitian_object_traits<stdex::mdspan<N, E, NL, NA>, tri>,
        internal::mdspan_base_object_traits<N, E, NL, NA>
    {};


    template<typename Nested, triangle_type storage_type>
    struct library_interface<hermitian_adapter<Nested, storage_type>>
    {
      template<typename Arg, triangle_type tri>
#ifdef __cpp_concepts
      static constexpr triangular_matrix<tri> decltype(auto)
#else
      static constexpr decltype(auto)
#endif
      to_triangular(Arg&& arg)
      {
        static_assert(tri != triangle_type::none);
        if constexpr (tri == storage_type or tri == triangle_type::diagonal)
          return to_triangular<tri>(std::forward<Arg>(arg).nested_object());
        else
          return to_triangular<tri>(conjugate_transpose(std::forward<Arg>(arg).nested_object()));
      };


    };


  }

}



#endif

