/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
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

namespace OpenKalman::internal
{
#ifdef __cpp_concepts
  template<indexible NestedObject, vector_space_descriptor...Vs> requires
    compatible_with_vector_space_descriptors<NestedObject, Vs...> and
    internal::not_more_fixed_than<NestedObject, Vs...> and internal::less_fixed_than<NestedObject, Vs...>
#else
  template<typename NestedObject, typename...Vs>
#endif
  struct FixedSizeAdapter : AdapterBase<FixedSizeAdapter<NestedObject, Vs...>, const NestedObject>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<NestedObject>);
    static_assert((vector_space_descriptor<Vs> and ...));
    static_assert(compatible_with_vector_space_descriptors<NestedObject, Vs...>);
    static_assert(internal::not_more_fixed_than<NestedObject, Vs...>);
    static_assert(internal::less_fixed_than<NestedObject, Vs...>);
#endif

    using Base = AdapterBase<FixedSizeAdapter, const NestedObject>;

  public:

    using Base::Base;


    /**
     * \brief Construct from a compatible indexible object.
     * \tparam Arg An \ref indexible object
     * \tparam Ds A set of optional \ref vector_space_descriptor objects,
     * which if included must be compatible with those of the FixedSizeAdapter
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg, vector_space_descriptor...Ds> requires
      (sizeof...(Ds) == 0 or (... and (equivalent_to<Ds, Vs> or (dynamic_vector_space_descriptor<Vs> and dynamic_vector_space_descriptor<Ds>)))) and
      (not fixed_size_adapter<Arg>) and std::constructible_from<Base, Arg&&>
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      compatible_with_vector_space_descriptors<Arg, Vs...> and (... and vector_space_descriptor<Ds>) and
      (sizeof...(Ds) == 0 or (... and (equivalent_to<Ds, Vs> or (dynamic_vector_space_descriptor<Vs> and dynamic_vector_space_descriptor<Ds>)))) and
      (not fixed_size_adapter<Arg>) and std::is_constructible_v<Base, Arg&&>, int> = 0>
#endif
    constexpr FixedSizeAdapter(Arg&& arg, const Ds&...) : Base {std::forward<Arg>(arg)} {}


    /**
     * \overload
     * \brief Construct from another FixedSizeAdapter.
     * \tparam Arg A \ref fixed_size_adapter
     * \tparam Ds A set of optional \ref vector_space_descriptor objects,
     * which if included must be compatible with those of the FixedSizeAdapter
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg, vector_space_descriptor...Ds> requires
      (sizeof...(Ds) == 0 or (... and (equivalent_to<Ds, Vs> or (dynamic_vector_space_descriptor<Vs> and dynamic_vector_space_descriptor<Ds>)))) and
      (fixed_size_adapter<Arg>) and std::constructible_from<Base, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      compatible_with_vector_space_descriptors<Arg, Vs...> and (... and vector_space_descriptor<Ds>) and
      (sizeof...(Ds) == 0 or (... and (equivalent_to<Ds, Vs> or (dynamic_vector_space_descriptor<Vs> and dynamic_vector_space_descriptor<Ds>)))) and
      (fixed_size_adapter<Arg>) and std::is_constructible_v<Base, nested_object_of_t<Arg&&>>, int> = 0>
#endif
    constexpr FixedSizeAdapter(Arg&& arg, const Ds&...) : Base {nested_object(std::forward<Arg>(arg))} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<compatible_with_vector_space_descriptors<Vs...> Arg> requires
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&>
#else
    template<typename Arg, std::enable_if_t<compatible_with_vector_space_descriptors<Arg, Vs...> and
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
    template<indexible Arg, vector_space_descriptor...Vs> requires (not fixed_size_adapter<Arg>)
#else
    template<typename Arg, typename...Vs, std::enable_if_t<indexible<Arg> and (... and vector_space_descriptor<Vs>) and
      (not fixed_size_adapter<Arg>), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Vs&...) -> FixedSizeAdapter<Arg, Vs...>;


#ifdef __cpp_concepts
    template<fixed_size_adapter Arg, vector_space_descriptor...Vs>
#else
    template<typename Arg, typename...Vs, std::enable_if_t<
      fixed_size_adapter<Arg> and (... and vector_space_descriptor<Vs>), int> = 0>
#endif
    FixedSizeAdapter(Arg&&, const Vs&...) ->
      FixedSizeAdapter<internal::remove_rvalue_reference_t<nested_object_of_t<Arg&&>>, Vs...>;


} // namespace OpenKalman::internal


#endif //OPENKALMAN_FIXEDSIZEADAPTER_HPP
