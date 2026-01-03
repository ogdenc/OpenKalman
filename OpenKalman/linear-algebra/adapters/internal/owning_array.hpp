/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref to_diagonal function.
 */

#ifndef OPENKALMAN_OWNING_ARRAY_HPP
#define OPENKALMAN_OWNING_ARRAY_HPP

#include "basics/basics.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/adapters/internal/AdapterBase.hpp"
#include "linear-algebra/adapters/interfaces/pass_through_interface.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename N>
    using nested_mdspan = std::decay_t<decltype(get_mdspan(std::declval<N>()))>;
  }


  /**
   * \brief An owning array that stores both an array and an associated mdspan.
   * \details This should only be used when an object is \ref indexible (i.e., get_mdspan is defined),
   * but there is no \ref interface::library_interface "library interface" for a particular operation.
   * \note This may be phased out and replaced with std::mdarray.
   */
#ifdef __cpp_concepts
  template<
    indexible Nested,
    pattern_collection_for<Nested> Extents = detail::nested_mdspan<Nested>::extents_type,
    typename LayoutPolicy = detail::nested_mdspan<Nested>::layout_type,
    typename AccessorPolicy = detail::nested_mdspan<Nested>::accessor_type>
#else
  template<
    typename Nested,
    typename Extents = detail::nested_mdspan<Nested>::extents_type,
    typename LayoutPolicy = detail::nested_mdspan<Nested>::layout_type,
    typename AccessorPolicy = detail::nested_mdspan<Nested>::accessor_type>
#endif
  struct owning_array
    : AdapterBase<owning_array<Nested, Extents, LayoutPolicy, AccessorPolicy>, Nested>
  {
  private:

    using Base = AdapterBase<owning_array, Nested>;
    using mdspan_type = stdex::mdspan<element_type_of_t<Nested>, Extents, LayoutPolicy, AccessorPolicy>;

  public:

    using extents_type = Extents;
    using layout_type = LayoutPolicy;
    using accessor_type = AccessorPolicy;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using element_type = element_type_of_t<Nested>;
    using value_type = std::remove_cv_t<element_type>;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using data_handle_type = typename accessor_type::data_handle_type;
    using reference = typename accessor_type::reference;

    /**
     * \brief Construct from an \ref indexible object and mdspan parameters.
     */
#ifdef __cpp_concepts
    template<indexible N> requires std::constructible_from<Nested, N&&>
#else
    template<typename N, std::enable_if_t<stdex::constructible_from<Nested, N&&>, int> = 0>
#endif
    constexpr
    owning_array(N&& n, const mapping_type& m, const accessor_type& a)
      : Base {std::forward<N>(n)},
        mdspan_ {get_mdspan(this->nested_object()).data_handle(), m, a} {}


    /**
     * \overload
     * \brief Construct from an \ref indexible object using that object's mdspan parameters.
     */
#ifdef __cpp_concepts
    template<indexible N> requires
      std::constructible_from<Nested, N&&> and
      std::constructible_from<mdspan_type, decltype(get_mdspan(std::declval<N>()))>
#else
    template<typename N, std::enable_if_t<
      stdex::constructible_from<Nested, N&&> and
      stdex::constructible_from<mdspan_type, decltype(get_mdspan(std::declval<N>()))>, int> = 0>
#endif
    constexpr
    owning_array(N&& n)
      : Base {std::forward<N>(n)},
        mdspan_ {get_mdspan(this->nested_object())} {}


    /**
     * \brief Default constructor.
     */
    constexpr owning_array() = default;


    /**
     * \brief Copy construct from another owning_array.
     */
#ifdef __cpp_concepts
    template<typename N, typename E, typename L, typename A> requires
      (not std::same_as<owning_array, owning_array<N, E, L, A>>) and
      std::constructible_from<Nested, N&&> and
      std::constructible_from<mdspan_type, const owning_array<N, E, L, A>&>
#else
    template<typename N, typename E, typename L, typename A, std::enable_if_t<
      (not std::same_as<owning_array, owning_array<N, E, L, A>>) and
      stdex::constructible_from<Nested, N&&> and
      stdex::constructible_from<mdspan_type, const owning_array<N, E, L, A>&>, int> = 0>
#endif
    constexpr
    owning_array(const owning_array<N, E, L, A>& arg)
      : Base {arg.nested_object()},
        mdspan_ {get_mdspan(arg)} {}


    /**
     * \brief Move construct from another owning_array.
     */
#ifdef __cpp_concepts
    template<typename N, typename E, typename L, typename A> requires
      (not std::same_as<owning_array, owning_array<N, E, L, A>>) and
      std::constructible_from<Nested, N&&> and
      std::constructible_from<mdspan_type, owning_array<N, E, L, A>&&>
#else
    template<typename N, typename E, typename L, typename A, std::enable_if_t<
      (not std::same_as<owning_array, owning_array<N, E, L, A>>) and
      stdex::constructible_from<Nested, N&&> and
      stdex::constructible_from<mdspan_type, owning_array<N, E, L, A>&&>, int> = 0>
#endif
    constexpr
    owning_array(owning_array<N, E, L, A>&& arg)
      : Base {std::move(arg).nested_object()},
        mdspan_ {get_mdspan(arg)} {}

  private:

    mdspan_type mdspan_;

    friend interface::object_traits<owning_array>;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<indexible N, typename M, typename A>
#else
  template<typename N, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
  owning_array(const N&, const M& m, const A& a)
    -> owning_array<N, typename M::extents_type, typename M::layout_type, A>;

}


////////////////////////
//     interfaces     //
////////////////////////

namespace OpenKalman::interface
{
  /**
   * \brief Interface traits for \ref owning_array
   */
  template<typename Nested, typename Mdspan>
  struct object_traits<internal::owning_array<Nested, Mdspan>>
    : pass_through_object_traits<internal::owning_array<Nested, Mdspan>, Nested>
  {
    static constexpr auto
    get_mdspan = [](auto&& t) -> decltype(auto) { return t.mdspan_; };
  };


  /**
   * \brief Library interface traits for \ref owning_array
   */
  template<typename Nested, typename Mdspan>
  struct library_interface<internal::owning_array<Nested, Mdspan>>
    : pass_through_library_interface<internal::owning_array<Nested, Mdspan>, Nested>
  {};

}


#endif
