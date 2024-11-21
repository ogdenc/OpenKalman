/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_VECTORSPACEADAPTER_HPP
#define OPENKALMAN_VECTORSPACEADAPTER_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<indexible NestedObject, vector_space_descriptor_collection Descriptors> requires
    internal::not_more_fixed_than<NestedObject, Descriptors> and (not internal::less_fixed_than<NestedObject, Descriptors>) and
    internal::maybe_same_shape_as_vector_space_descriptors<NestedObject, Descriptors>
#else
  template<typename NestedObject, typename Descriptors>
#endif
  struct VectorSpaceAdapter : internal::AdapterBase<VectorSpaceAdapter<NestedObject, Descriptors>, NestedObject>
  {
  private:

#ifndef __cpp_concepts
    static_assert(indexible<NestedObject>);
    static_assert(vector_space_descriptor_collection<Descriptors>);
    static_assert(internal::not_more_fixed_than<NestedObject, Descriptors>);
    static_assert(not internal::less_fixed_than<NestedObject, Descriptors>);
    static_assert(internal::maybe_same_shape_as_vector_space_descriptors<NestedObject, Descriptors>);
#endif

    using Base = internal::AdapterBase<VectorSpaceAdapter, NestedObject>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr VectorSpaceAdapter() requires std::default_initializable<Base> and
      static_vector_space_descriptor_tuple<Descriptors>
#else
    template<typename B = Base, std::enable_if_t<std::is_default_constructible<B>::value
      and static_vector_space_descriptor_tuple<Descriptors>, int> = 0>
    constexpr VectorSpaceAdapter()
#endif
      : Base {}, my_descriptors{} {}


    /**
     * \brief Construct from a compatible indexible object.
     * \tparam Arg An \ref indexible object. Any of its vector space descriptors will be overwritten.
     * \param descriptors A set of \ref vector_space_descriptor objects
     */
#ifdef __cpp_concepts
    template<internal::maybe_same_shape_as_vector_space_descriptors<Descriptors> Arg> requires
      (not internal::vector_space_adapter<Arg>) and std::constructible_from<Base, Arg&&> 
#else
    template<typename Arg, typename...Ds, std::enable_if_t<
      internal::maybe_same_shape_as_vector_space_descriptors<Arg, Descriptors> and
      (not internal::vector_space_adapter<Arg>) and std::is_constructible_v<Base, Arg&&>, int> = 0>
#endif
    constexpr VectorSpaceAdapter(Arg&& arg, const std::decay_t<Descriptors>& descriptors)
      : Base {std::forward<Arg>(arg)}, my_descriptors {descriptors} {}


    /**
     * \overload
     * \brief Construct from another VectorSpaceAdapter, overwriting its vector space descriptors.
     */
#ifdef __cpp_concepts
    template<internal::maybe_same_shape_as_vector_space_descriptors<Descriptors> Arg> requires
      internal::vector_space_adapter<Arg> and std::constructible_from<Base, nested_object_of_t<Arg&&>>
#else
    template<typename Arg, std::enable_if_t<
      internal::maybe_same_shape_as_vector_space_descriptors<Arg, Descriptors> and
      internal::maybe_same_shape_as_vector_space_descriptors<Arg, Descriptors> and
      internal::vector_space_adapter<Arg> and std::is_constructible_v<Base, typename nested_object_of<Arg&&>::type>, int> = 0>
#endif
    constexpr VectorSpaceAdapter(Arg&& arg, const std::decay_t<Descriptors>& descriptors)
      : Base {nested_object(std::forward<Arg>(arg))}, my_descriptors {descriptors} {}


    /**
     * \brief Assign from another compatible indexible object.
     */
#ifdef __cpp_concepts
    template<internal::maybe_same_shape_as_vector_space_descriptors<Descriptors> Arg> requires
      (not std::is_base_of_v<VectorSpaceAdapter, std::decay_t<Arg>>) and 
      std::assignable_from<std::add_lvalue_reference_t<NestedObject>, Arg&&> and
      requires(Arg&& arg) { {count_indices(arg)} -> value::static_index; } and
      requires(Descriptors my_descriptors, Arg&& arg) { my_descriptors = all_vector_space_descriptors(arg); }
#else
    template<typename Arg, std::enable_if_t<
      (not std::is_base_of_v<VectorSpaceAdapter, std::decay_t<Arg>>) and 
      internal::maybe_same_shape_as_vector_space_descriptors<Arg, Descriptors> and
      std::is_assignable_v<std::add_lvalue_reference_t<NestedObject>, Arg&&> and
      value::static_index<decltype(count_indices(std::declval<Arg&&>()))> and
      std::is_assignable_v<Descriptors, decltype(all_vector_space_descriptors(std::declval<Arg&&>()))>, int> = 0>
#endif
    constexpr VectorSpaceAdapter& operator=(Arg&& arg)
    {
      my_descriptors = all_vector_space_descriptors(arg);
      Base::operator=(std::forward<Arg>(arg));
      return *this;
    }


    /// Increment from another VectorSpaceAdapter.
#ifdef __cpp_concepts
    auto& operator+=(const VectorSpaceAdapter& other) requires
      requires(Base& base, const VectorSpaceAdapter& v) { base += v.nested_object(); }
#else
    template<typename B = Base, typename = std::void_t<
      decltype(std::declval<Base&>() += std::declval<const VectorSpaceAdapter&>().nested_object())>>
    auto& operator+=(const VectorSpaceAdapter& other)
#endif
    {
      Base::operator+=(other.nested_object());
      return *this;
    }


    /// Decrement from another VectorSpaceAdapter.
#ifdef __cpp_concepts
    auto& operator-=(const VectorSpaceAdapter& other) requires
      requires(Base& base, const VectorSpaceAdapter& v) { base -= v.nested_object(); }
#else
    template<typename B = Base, typename = std::void_t<
      decltype(std::declval<Base&>() -= std::declval<const VectorSpaceAdapter&>().nested_object())>>
    auto& operator-=(const VectorSpaceAdapter& other)
#endif
    {
      Base::operator-=(other.nested_object());
      return *this;
    }


  protected:

    std::decay_t<Descriptors> my_descriptors;

    friend struct interface::indexible_object_traits<VectorSpaceAdapter>;
    friend struct interface::library_interface<VectorSpaceAdapter>;

  }; // struct VectorSpaceAdapter


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
    template<indexible Arg, vector_space_descriptor_collection Descriptors> requires (not internal::vector_space_adapter<Arg>)
#else
    template<typename Arg, typename...Vs, std::enable_if_t<indexible<Arg> and vector_space_descriptor_collection<Descriptors> and
      (not internal::vector_space_adapter<Arg>), int> = 0>
#endif
    VectorSpaceAdapter(Arg&&, Descriptors&&)
      -> VectorSpaceAdapter<Arg, std::decay_t<Descriptors>>;


#ifdef __cpp_concepts
    template<internal::vector_space_adapter Arg, vector_space_descriptor_collection Descriptors>
#else
    template<typename Arg, typename...Vs, std::enable_if_t<
      internal::vector_space_adapter<Arg> and vector_space_descriptor_collection<Descriptors>, int> = 0>
#endif
    VectorSpaceAdapter(Arg&&, Descriptors&&)
      -> VectorSpaceAdapter<internal::remove_rvalue_reference_t<nested_object_of_t<Arg&&>>, std::decay_t<Descriptors>>;


} // namespace OpenKalman


#endif //OPENKALMAN_VECTORSPACEADAPTER_HPP
