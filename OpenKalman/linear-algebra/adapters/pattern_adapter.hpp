/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref pattern_adapter.
 */

#ifndef OPENKALMAN_PATTERN_ADAPTER_HPP
#define OPENKALMAN_PATTERN_ADAPTER_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/concepts/pattern_collection_for.hpp"
#include "linear-algebra/adapters/internal/AdapterBase.hpp"
#include "linear-algebra/adapters/interfaces/pass_through_interface.hpp"

namespace OpenKalman
{
  /**
   * \brief An adapter that attaches a \ref patterns::pattern_collection "pattern_collection" to an \ref indexible object.
   * \details Any vector space descriptors associated with the nested object are effectively overwritten.
   * The adapter can be owning or non-owning, depending on whether Nested is an lvalue reference.
   */
#ifdef __cpp_concepts
  template<indexible Nested, pattern_collection_for<Nested> PatternCollection>
  requires std::same_as<PatternCollection, std::decay_t<PatternCollection>>
#else
  template<typename Nested, typename PatternCollection>
#endif
  struct pattern_adapter : internal::AdapterBase<pattern_adapter<Nested, PatternCollection>, Nested>
  {
  private:

#ifndef __cpp_concepts
    static_assert(pattern_collection_for<PatternCollection, Nested>);
#endif

    using Base = internal::AdapterBase<pattern_adapter, Nested>;

  public:

    /**
     * \brief Construct from an \ref indexible object and a \ref patterns::pattern_collection "pattern_collection".
     */
#ifdef __cpp_concepts
    template<indexible Arg, patterns::pattern_collection P> requires
      std::constructible_from<Nested, Arg&&> and
      std::constructible_from<PatternCollection, P&&>
#else
    template<typename Arg, typename P, std::enable_if_t<
      stdex::constructible_from<Nested, Arg&&> and
      stdex::constructible_from<PatternCollection, P&&>, int> = 0>
#endif
    constexpr
    pattern_adapter(Arg&& arg, P&& p) : Base {std::forward<Arg>(arg)}, patt_ {std::forward<P>(p)}
    {
      if constexpr(not pattern_collection_for<PatternCollection, Nested, applicability::guaranteed>)
      {
        // Note: this could be expensive in some circumstances:
        assert(not patterns::compare_pattern_collections(to_extents(patt_), get_pattern_collection(this->nested_object())));
      }
    }


    /**
     * \brief Default constructor.
     */
    constexpr pattern_adapter() = default;


    /**
     * \brief Get the associated \ref patterns::pattern_collection "pattern_collection".
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self>
    constexpr decltype(auto) pattern_collection(this Self&& self)
    {
      return std::forward<Self>(self).patt_;
    }
#else
    constexpr PatternCollection& pattern_collection() & { return patt_; }

    /// \overload
    constexpr const PatternCollection& pattern_collection() const & { return patt_; }

    /// \overload
    constexpr PatternCollection&& pattern_collection() && { return std::move(*this).patt_; }

    /// \overload
    constexpr const PatternCollection&& pattern_collection() const && { return std::move(*this).patt_; }
#endif

  private:

    PatternCollection patt_;

  };


  /**
   * \brief Deduction guide
   */
#ifdef __cpp_concepts
  template<indexible Arg, pattern_collection_for<Arg> P>
#else
  template<typename Arg, typename P, std::enable_if_t<indexible<Arg> and pattern_collection_for<P, Arg>, int> = 0>
#endif
  pattern_adapter(Arg&&, P&&) -> pattern_adapter<Arg, stdex::remove_cvref_t<P>>;


  ///////////////////////
  //     interface     //
  ///////////////////////

  namespace interface
  {
    /**
     * \brief Interface traits for \ref pattern_adapter
     */
    template<typename Nested, typename PatternCollection>
    struct object_traits<pattern_adapter<Nested, PatternCollection>>
      : pass_through_object_traits<pattern_adapter<Nested, PatternCollection>, Nested>
    {
      static constexpr auto
      get_pattern_collection = [](auto&& t) -> decltype(auto)
      { return std::forward<decltype(t)>(t).pattern_collection(); };
    };


    /**
     * \brief Library interface traits for \ref pattern_adapter
     */
    template<typename Nested, typename PatternCollection>
    struct library_interface<pattern_adapter<Nested, PatternCollection>>
      : pass_through_library_interface<pattern_adapter<Nested, PatternCollection>, Nested>
    {};

  }


}


#endif
