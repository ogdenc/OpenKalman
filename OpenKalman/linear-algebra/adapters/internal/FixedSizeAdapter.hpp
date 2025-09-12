/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of FixedSizeAdapter.
 */

#ifndef OPENKALMAN_FIXEDSIZEADAPTER_HPP
#define OPENKALMAN_FIXEDSIZEADAPTER_HPP

#include "basics/basics.hpp"

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedObject, pattern_collection Descriptors> requires
    compatible_with_vector_space_descriptor_collection<NestedObject, Descriptors> and
    internal::not_more_fixed_than<NestedObject, Descriptors> and internal::less_fixed_than<NestedObject, Descriptors>
#else
  template<typename NestedObject, typename Descriptors>
#endif
  struct FixedSizeAdapter : AdapterBase<FixedSizeAdapter<NestedObject, Descriptors>, const NestedObject>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<NestedObject>);
    static_assert(pattern_collection<Descriptors>);
    static_assert(compatible_with_vector_space_descriptor_collection<NestedObject, Descriptors>);
    static_assert(internal::not_more_fixed_than<NestedObject, Descriptors>);
    static_assert(internal::less_fixed_than<NestedObject, Descriptors>);
#endif

    using Base = AdapterBase<FixedSizeAdapter, const NestedObject>;

  public:

    using Base::Base;


    /**
     * \brief Construct from a compatible indexible object.
     * \tparam Arg An \ref indexible object
     * \tparam D A compatible \ref pattern_collection
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptor_collection<Descriptors> Arg> requires
      std::constructible_from<NestedObject, Arg&&> and (not fixed_size_adapter<Arg>)
#else
    template<typename Arg, std::enable_if_t<
      compatible_with_vector_space_descriptors<Arg, Descriptors> and stdcompat::constructible_from<NestedObject, Arg&&> and
        (not fixed_size_adapter<Arg>), int> = 0>
#endif
    constexpr FixedSizeAdapter(Arg&& arg, const Descriptors&) : Base {std::forward<Arg>(arg)} {}


    /**
     * \overload
     * \tparam Ds A set of optional \ref coordinates::pattern objects,
     * which if included must be compatible with those of the FixedSizeAdapter
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptor_collection<Descriptors> Arg, coordinates::pattern...Ds> requires
      (sizeof...(Ds) == 0 or std::same_as<std::tuple<Ds...>, Descriptors>) and
      std::constructible_from<NestedObject, Arg&&> and (not fixed_size_adapter<Arg>)
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      compatible_with_vector_space_descriptors<Arg, Vs...> and (... and coordinates::pattern<Ds>) and
      std::is_same_v<std::tuple<Ds...>, Descriptors> and
      stdcompat::constructible_from<NestedObject, Arg&&> and (not fixed_size_adapter<Arg>), int> = 0>
#endif
    constexpr FixedSizeAdapter(Arg&& arg, const Ds&...) : Base {std::forward<Arg>(arg)} {}


    /**
     * \overload
     * \brief Construct from another FixedSizeAdapter.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptor_collection<Descriptors> Arg> requires
      std::constructible_from<NestedObject, nested_object_of_t<Arg&&>> and fixed_size_adapter<Arg>
#else
    template<typename Arg, std::enable_if_t<
      compatible_with_vector_space_descriptors<Arg, Descriptors> and
      stdcompat::constructible_from<NestedObject, nested_object_of_t<Arg&&>> and fixed_size_adapter<Arg>, int> = 0>
#endif
    constexpr FixedSizeAdapter(Arg&& arg, const Descriptors&...) : Base {nested_object(std::forward<Arg>(arg))} {}


    /**
     * \overload
     * \brief Construct from another FixedSizeAdapter.
     * \tparam Arg A \ref fixed_size_adapter
     * \tparam Ds A set of optional \ref coordinates::pattern objects,
     * which if included must be compatible with those of the FixedSizeAdapter
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptor_collection<Descriptors> Arg, coordinates::pattern...Ds> requires
      (sizeof...(Ds) == 0 or std::same_as<std::tuple<Ds...>, Descriptors>) and
      std::constructible_from<NestedObject, nested_object_of_t<Arg&&>> and fixed_size_adapter<Arg>
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      compatible_with_vector_space_descriptors<Arg, Vs...> and (... and coordinates::pattern<Ds>) and
      std::is_same_v<std::tuple<Ds...>, Descriptors> and
      stdcompat::constructible_from<NestedObject, nested_object_of_t<Arg&&>> and fixed_size_adapter<Arg>, int> = 0>
#endif
    constexpr FixedSizeAdapter(Arg&& arg, const Ds&...) : Base {nested_object(std::forward<Arg>(arg))} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptor_collection<Descriptors> Arg> requires
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<compatible_with_vector_space_descriptor_collection<Arg, Descriptors> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, Arg&&>, int> = 0>
#endif
    constexpr FixedSizeAdapter& operator=(Arg&& arg)
    {
      Base::operator=(std::forward<Arg>(arg));
      return *this;
    }


    /**
     * \brief Get the nested object.
     */
    constexpr decltype(auto) nested_object() const & { return Base::nested_object(); }

    /// \overload
    constexpr decltype(auto) nested_object() const && { return std::move(static_cast<const Base&&>(*this)).nested_object(); }

  };


  // ------------------ //
  //  Deduction Guides  //
  // ------------------ //

#ifdef __cpp_concepts
    template<indexible Arg, pattern_collection Descriptors> requires (not fixed_size_adapter<Arg>)
#else
    template<typename Arg, typename...Vs, std::enable_if_t<indexible<Arg> and pattern_collection<Descriptors> and
      (not fixed_size_adapter<Arg>), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Descriptors&) -> FixedSizeAdapter<Arg, Descriptors>;


#ifdef __cpp_concepts
    template<indexible Arg, coordinates::pattern...Vs> requires (not fixed_size_adapter<Arg>)
#else
    template<typename Arg, typename...Vs, std::enable_if_t<indexible<Arg> and (... and coordinates::pattern<Vs>) and
      (not fixed_size_adapter<Arg>), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Vs&...) -> FixedSizeAdapter<Arg, std::tuple<Vs...>>;


#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, pattern_collection Descriptors>
#else
    template<typename Arg, typename Descriptors, std::enable_if_t<fixed_size_adapter<Arg> and pattern_collection<Descriptors>, int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Descriptors&) ->
      FixedSizeAdapter<internal::remove_rvalue_reference_t<nested_object_of_t<Arg&&>>, Descriptors>;


#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, coordinates::pattern...Vs>
#else
    template<typename Arg, typename...Vs, std::enable_if_t<
      fixed_size_adapter<Arg> and (... and coordinates::pattern<Vs>), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Vs&...) ->
      FixedSizeAdapter<internal::remove_rvalue_reference_t<nested_object_of_t<Arg&&>>, std::tuple<Vs...>>;


}


#endif
