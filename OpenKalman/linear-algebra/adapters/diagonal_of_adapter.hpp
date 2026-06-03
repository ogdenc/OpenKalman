/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref diagonal_of_adapter
 */

#ifndef OPENKALMAN_DIAGONAL_OF_ADAPTER_HPP
#define OPENKALMAN_DIAGONAL_OF_ADAPTER_HPP

#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/adapters/internal/adapter_base.hpp"
#include "linear-algebra/interfaces/stl/mdspan-object.hpp"
#include "linear-algebra/interfaces/stl/layout_diagonal_of.hpp"

namespace OpenKalman
{

#ifdef __cpp_concepts
  template<indexible Nested>
#else
  template<typename Nested>
#endif
  struct diagonal_of_adapter
    : OpenKalman::internal::adapter_base<diagonal_of_adapter<Nested>, Nested>
  {

#ifndef __cpp_concepts
    static_assert(indexible<Nested>);
#endif

  private:

    using Base = OpenKalman::internal::adapter_base<diagonal_of_adapter, Nested>;

  public:

    /**
     * \brief Construct from the nested type.
     */
#ifdef __cpp_concepts
    template<indexible Arg> requires std::constructible_from<Nested, Arg&&>
#else
    template<typename Arg, typename P, std::enable_if_t<stdex::constructible_from<Nested, Arg&&>, int> = 0>
#endif
    constexpr
    diagonal_of_adapter(Arg&& arg) : Base {std::forward<Arg>(arg)}
    {}


    /**
     * \brief Default constructor.
     */
    constexpr diagonal_of_adapter() = default;

  };


  // ------------------------------ //
  //        Deduction guide         //
  // ------------------------------ //

  /**
   * \brief Deduce diagonal_of_adapter Nested from its constructor argument.
   */
#ifdef __cpp_concepts
  template<indexible Arg>
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  explicit diagonal_of_adapter(Arg&&) -> diagonal_of_adapter<Arg>;


  // ------------------------- //
  //        Interfaces         //
  // ------------------------- //

  namespace interface
  {
    template<typename Nested>
    struct object_traits<diagonal_of_adapter<Nested>>
    {
      static const bool is_specialized = true;

      static constexpr auto
      get_mdspan = [](auto&& t)
      {
        auto p = patterns::views::diagonal_of(get_pattern_collection(t.nested_object()));
        auto n = OpenKalman::get_mdspan(t.nested_object());
        using N = std::decay_t<decltype(n)>;
        using nested_extents_type = typename N::extents_type;
        using nested_layout = typename N::layout_type;

        using layout_type = layout_diagonal_of<nested_layout, nested_extents_type>;
        using extents_type = decltype(patterns::to_extents(p));
        using mapping_type = typename layout_type::template mapping<extents_type>;

        return stdex::mdspan(
          n.data_handle(),
          mapping_type {n.mapping(), patterns::to_extents(p)},
          n.accessor());
      };


      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      {
        return patterns::views::diagonal_of(OpenKalman::get_pattern_collection(t.nested_object()));
      };


#ifdef __cpp_concepts
      template<typename T> requires get_constant_defined_for<Nested>
#else
      template<typename T, bool Enable = true, std::enable_if_t<Enable and get_constant_defined_for<Nested>, int> = 0>
#endif
      static constexpr auto
      get_constant(T&& t)
      {
        return constant_value(std::forward<T>(t).nested_object());
      }

    };


    template<typename N, typename E, typename NL, typename NE, typename A>
    struct object_traits<stdex::mdspan<N, E, layout_diagonal_of<NL, NE>, A>>
      : internal::mdspan_base_object_traits<N, NE, NL, A>
    {
      static const bool is_specialized = true;
    };


    template<typename N, typename E, typename NL, typename NE, typename NA>
    struct object_traits<stdex::mdspan<N, E, layout_diagonal_of<NL, NE>, stdex::linalg::conjugated_accessor<NA>>>
      : internal::mdspan_base_object_traits<N, NE, NL, stdex::linalg::conjugated_accessor<NA>>
    {
      static const bool is_specialized = true;
    };


  }

}



#endif
