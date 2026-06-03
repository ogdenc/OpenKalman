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
 * \brief Definitions for triangular_adapter
 */

#ifndef OPENKALMAN_TRIANGULAR_ADAPTER_HPP
#define OPENKALMAN_TRIANGULAR_ADAPTER_HPP

#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/triangular_accessor.hpp"
#include "linear-algebra/interfaces/stl/layout_triangle_partition.hpp"

namespace OpenKalman
{

  /**
   * \brief An adapter to create a \ref triangular_matrix from a general matrix.
   * \details Elements outside the triangle are zero. The matrix may be a diagonal matrix if tri is triangle_type::diagonal.
   * \tparam Nested A nested matrix on which the triangular matrix is based. Components above or below the diagonal
   * (or both) are ignored and will read as zero.
   * \tparam tri The triangle_type (\ref triangle_type::lower "lower", \ref triangle_type::upper "upper", or
   * \ref triangle_type::diagonal "diagonal") in which the data is stored.
   */
#ifdef __cpp_concepts
  template<indexible Nested, triangle_type tri> requires (tri != triangle_type::none)
#else
  template<typename Nested, triangle_type tri>
#endif
  struct triangular_adapter : OpenKalman::internal::adapter_base<triangular_adapter<Nested, tri>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<Nested>);
    static_assert(tri != triangle_type::none);
#endif

    using Base = OpenKalman::internal::adapter_base<triangular_adapter, Nested>;

  public:

    using Base::Base;

  };


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    namespace detail
    {
      template<typename Nested, triangle_type tri>
      struct triangular_object_traits
      {
        static const bool is_specialized = true;

        static constexpr triangle_type
        triangle_type_value = tri * triangle_type_of_v<Nested>;

        template<std::size_t N, applicability b>
        static constexpr bool
        is_square = square_shaped<Nested, N, b>;

      };
    }


    template<typename Nested, triangle_type tri>
    struct object_traits<triangular_adapter<Nested, tri>>
      : detail::triangular_object_traits<stdex::remove_cvref_t<Nested>, tri>
    {
      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        auto n = OpenKalman::get_mdspan(std::forward<decltype(t)>(t).nested_object());
        using N = std::decay_t<decltype(n)>;
        using nested_extents_type = typename N::extents_type;
        using nested_layout = typename N::layout_type;
        using nested_accessor = typename N::accessor_type;
        auto nested_m = n.mapping();

        using layout_type = layout_triangle_partition<nested_layout, tri>;
        using mapping_type = typename layout_type::template mapping<nested_extents_type>;
        std::size_t partition_offset = nested_m.required_span_size();

        using accessor_type = triangular_accessor<nested_accessor>;
        using data_handle_type = typename accessor_type::data_handle_type;

        return stdex::mdspan(
          data_handle_type {n.data_handle(), partition_offset},
          mapping_type {nested_m},
          accessor_type {n.accessor()});
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return OpenKalman::get_pattern_collection(std::forward<decltype(t)>(t).nested_object());
      };


#ifdef __cpp_concepts
      template<typename T> requires get_constant_defined_for<Nested>
#else
      template<typename T, bool Enable = true, std::enable_if_t<Enable and get_constant_defined_for<Nested>, int> = 0>
#endif
      static constexpr auto
      get_constant(T&& t)
      {
        return object_traits<stdex::remove_cvref_t<Nested>>::get_constant(std::forward<T>(t).nested_object());
      };

    };


    template<typename N, typename E, typename NL, triangle_type tri, typename NA>
    struct object_traits<stdex::mdspan<N, E, layout_triangle_partition<NL, tri>, triangular_accessor<NA>>>
      : detail::triangular_object_traits<stdex::mdspan<N, E, NL, NA>, tri>,
        internal::mdspan_base_object_traits<N, E, NL, NA>
    {};

  }

}


#endif

